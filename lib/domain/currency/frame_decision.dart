import 'package:bigash/domain/currency/yemeni_currency_label_mapper.dart';

/// Pure, side-effect-free decision logic for a single inference frame and for
/// the rolling vote. Extracted from `DetectionCubit` so the safety-critical
/// path (model output -> "is this a confident Yemeni denomination?") can be
/// unit-tested with no camera/TTS/vibration/wakelock plugins in the loop.
///
/// Everything here is deterministic and takes its inputs explicitly (including
/// the clock), so tests pin the exact behaviour that decides what gets spoken.

/// Collapses the raw classifier top-K (whose keys are design sub-variants such
/// as `y100_new` / `y100_old`) into PARENT denominations, summing the
/// probability mass of each group, sorted by probability descending.
///
/// This must run BEFORE any confidence/margin gate: the model deliberately
/// splits y100 -> {y100_new, y100_old} and y200 -> {y200_new, y200_old}, so a
/// genuine 100-riyal bill can spread its mass across the two sub-designs
/// (e.g. 0.31 / 0.27). Comparing the raw sub-variants would read that as a
/// near-tie and suppress the announcement; grouping first restores the real
/// 0.58 confidence for the y100 denomination. Ties break on the key name so
/// the ordering is deterministic (no HashMap-iteration dependence).
List<MapEntry<String, double>> groupTopK(
  List<MapEntry<String, double>> rawTopK,
) {
  final Map<String, double> sums = <String, double>{};
  for (final MapEntry<String, double> e in rawTopK) {
    final String group = denominationGroupForVoting(
      normalizeYemeniClassifierLabel(e.key),
    );
    sums[group] = (sums[group] ?? 0) + e.value;
  }
  final List<MapEntry<String, double>> out = sums.entries.toList()
    ..sort((MapEntry<String, double> a, MapEntry<String, double> b) {
      final int byValue = b.value.compareTo(a.value);
      return byValue != 0 ? byValue : a.key.compareTo(b.key);
    });
  return out;
}

/// What a single frame is — before voting smooths across frames.
enum FrameVerdict {
  /// A confident Yemeni denomination that clears every gate (incl. OOD).
  currency,

  /// Decisively NOT a Yemeni bill (the `other` mass dominates).
  unknown,

  /// Ambiguous / low confidence — guide the user, do not vote either way.
  uncertain,
}

class FrameAssessment {
  /// The grouped denomination key (e.g. `y100`) when [verdict] is
  /// [FrameVerdict.currency]; `other` when [FrameVerdict.unknown]; empty
  /// when [FrameVerdict.uncertain].
  final String key;
  final FrameVerdict verdict;

  /// Probability mass behind [key] (grouped). For logging/diagnostics.
  final double confidence;

  const FrameAssessment(this.verdict, this.key, this.confidence);
}

/// Tunable gates for [classifyFrame]. Defaults mirror the field-tuned values.
class DecisionThresholds {
  /// Minimum grouped probability for a currency denomination to be eligible.
  /// Calibrated to the retrained model (lighter MixUp + label smoothing): real
  /// bills now land at ~0.88-0.95 grouped confidence, while borderline OOD
  /// frames sit near the old 0.35 floor — so 0.50 rejects those without
  /// touching genuine bills. Raise toward 0.6 only if field false positives
  /// persist; lower if real bills get rejected (measure with tools/eval_model.py).
  final double currencyThreshold;

  /// Minimum grouped mass for the `other`/unknown verdict.
  final double otherThreshold;

  /// The top currency denomination must beat the runner-up DENOMINATION by
  /// this margin (computed on grouped mass, never on raw sub-variants).
  final double voteMargin;

  /// OUT-OF-DISTRIBUTION gate (margin form): the top currency denomination
  /// must beat the `other` mass by this margin. Catches foreign notes that
  /// make `other` COMPETE with the top Yemeni class.
  final double oodMargin;

  /// OUT-OF-DISTRIBUTION gate (ceiling form): refuse to announce a denomination
  /// while `other` carries more than this much mass, EVEN IF a currency class
  /// is confidently leading. A genuine Yemeni bill drives `other` to near zero;
  /// a meaningful `other` signal (e.g. ~0.13 observed on an Indian 5-rupee that
  /// the model confidently — and wrongly — called y200 at 0.60) means the model
  /// sees non-Yemeni features and must not commit. Biases toward "unknown" over
  /// a wrong announcement, per the project's safety-first stance. Field-tunable:
  /// raise it if real worn bills get rejected too often. NOTE: this only REDUCES
  /// such errors; the definitive fix is retraining with the confuser in `other`.
  final double otherCeiling;

  const DecisionThresholds({
    this.currencyThreshold = 0.50,
    this.otherThreshold = 0.28,
    this.voteMargin = 0.10,
    this.oodMargin = 0.15,
    this.otherCeiling = 0.12,
  });
}

/// Classifies one frame's grouped distribution into a [FrameAssessment].
/// [groupedTopK] MUST already be grouped (see [groupTopK]) and sorted desc.
FrameAssessment classifyFrame(
  List<MapEntry<String, double>> groupedTopK,
  DecisionThresholds t,
) {
  if (groupedTopK.isEmpty) {
    return const FrameAssessment(FrameVerdict.uncertain, '', 0);
  }

  double otherProb = 0;
  final List<MapEntry<String, double>> currency = <MapEntry<String, double>>[];
  for (final MapEntry<String, double> e in groupedTopK) {
    if (isIgnoredYemeniCurrencyLabel(e.key)) {
      otherProb += e.value;
    } else {
      currency.add(e);
    }
  }

  if (currency.isEmpty) {
    // Only non-currency mass is present.
    if (otherProb >= t.otherThreshold) {
      return FrameAssessment(FrameVerdict.unknown, 'other', otherProb);
    }
    return const FrameAssessment(FrameVerdict.uncertain, '', 0);
  }

  final double topCurrency = currency[0].value;
  final double secondCurrency = currency.length > 1 ? currency[1].value : 0.0;

  final bool confident = topCurrency >= t.currencyThreshold;
  final bool beatsOtherDenomination =
      (topCurrency - secondCurrency) >= t.voteMargin;
  final bool beatsOther = (topCurrency - otherProb) >= t.oodMargin;
  final bool otherLowEnough = otherProb <= t.otherCeiling;

  if (confident && beatsOtherDenomination && beatsOther && otherLowEnough) {
    return FrameAssessment(FrameVerdict.currency, currency[0].key, topCurrency);
  }

  // Not a confident Yemeni denomination. If `other` clearly leads, say so;
  // otherwise it is simply ambiguous and the user should reposition.
  if (otherProb >= t.otherThreshold &&
      (otherProb - topCurrency) >= t.voteMargin) {
    return FrameAssessment(FrameVerdict.unknown, 'other', otherProb);
  }

  return const FrameAssessment(FrameVerdict.uncertain, '', 0);
}

/// Time-bounded majority voter. An announcement requires a label to be both
/// the plurality winner AND a strict majority of the votes currently in the
/// window (count*2 > total), with a minimum count floor — replacing the old
/// "3 of a possibly-stale 10" plurality. Votes older than [maxAge] are pruned
/// so a slow scan cannot fire on minutes-old frames, and ties break on the
/// label name so the winner is deterministic.
class MajorityVoter {
  MajorityVoter({
    this.windowSize = 10,
    this.minMatches = 3,
    this.maxAge = const Duration(seconds: 5),
  });

  final int windowSize;
  final int minMatches;
  final Duration maxAge;

  final List<MapEntry<String, DateTime>> _votes =
      <MapEntry<String, DateTime>>[];

  int get length => _votes.length;

  void add(String label, DateTime now) {
    _prune(now);
    _votes.add(MapEntry<String, DateTime>(label, now));
    while (_votes.length > windowSize) {
      _votes.removeAt(0);
    }
  }

  void clear() => _votes.clear();

  void _prune(DateTime now) {
    _votes.removeWhere(
      (MapEntry<String, DateTime> v) => now.difference(v.value) > maxAge,
    );
  }

  /// The label that is both the plurality AND a strict majority of the current
  /// (pruned) window and reaches [minMatches], or null. Deterministic.
  String? winner(DateTime now) {
    _prune(now);
    if (_votes.length < minMatches) {
      return null;
    }
    final Map<String, int> counts = <String, int>{};
    for (final MapEntry<String, DateTime> v in _votes) {
      counts[v.key] = (counts[v.key] ?? 0) + 1;
    }
    final List<MapEntry<String, int>> ranked = counts.entries.toList()
      ..sort((MapEntry<String, int> a, MapEntry<String, int> b) {
        final int byCount = b.value.compareTo(a.value);
        return byCount != 0 ? byCount : a.key.compareTo(b.key);
      });
    final MapEntry<String, int> best = ranked.first;
    if (best.value >= minMatches && best.value * 2 > _votes.length) {
      return best.key;
    }
    return null;
  }
}
