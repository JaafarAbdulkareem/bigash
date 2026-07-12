// Unit tests for the safety-critical decision core: grouping the classifier's
// raw sub-variant probabilities, the per-frame confidence/margin/OOD gates, and
// the time-bounded majority voter. These are the pure functions that decide
// what denomination (if any) gets spoken — so they are tested directly, with no
// camera/TTS/vibration plugins in the loop.

import 'package:flutter_test/flutter_test.dart';
import 'package:bigash/domain/currency/frame_decision.dart';

List<MapEntry<String, double>> e(Map<String, double> m) => m.entries.toList();

void main() {
  group('groupTopK', () {
    test(
      'sums the new/old sub-variants of one denomination into the parent',
      () {
        // A worn 100 bill splits its mass across its two designs.
        final List<MapEntry<String, double>> grouped =
            groupTopK(<MapEntry<String, double>>[
              const MapEntry<String, double>('y100_new', 0.31),
              const MapEntry<String, double>('y100_old', 0.27),
              const MapEntry<String, double>('other', 0.09),
            ]);
        expect(grouped.first.key, 'y100');
        expect(grouped.first.value, closeTo(0.58, 1e-9));
        expect(grouped[1].key, 'other');
      },
    );

    test('strips index prefixes and is deterministic on ties', () {
      final List<MapEntry<String, double>> grouped =
          groupTopK(<MapEntry<String, double>>[
            const MapEntry<String, double>('4 y200_new', 0.20),
            const MapEntry<String, double>('5 y200_old', 0.20),
            const MapEntry<String, double>('7 y50', 0.20),
          ]);
      // y200 (0.40) leads; y50 (0.20) second. Deterministic order.
      expect(grouped.first.key, 'y200');
      expect(grouped.first.value, closeTo(0.40, 1e-9));
      expect(grouped[1].key, 'y50');
    });
  });

  group('classifyFrame', () {
    const DecisionThresholds t = DecisionThresholds();

    test('REGRESSION: a real y100 split across its two designs is announced, '
        'not stuck as a near-tie (margin must run on grouped mass)', () {
      final FrameAssessment a = classifyFrame(
        groupTopK(<MapEntry<String, double>>[
          const MapEntry<String, double>('y100_new', 0.31),
          const MapEntry<String, double>('y100_old', 0.27),
          const MapEntry<String, double>('other', 0.09),
        ]),
        t,
      );
      expect(a.verdict, FrameVerdict.currency);
      expect(a.key, 'y100');
    });

    test('a confident, clearly-leading denomination is currency', () {
      final FrameAssessment a = classifyFrame(
        e(<String, double>{'y500': 0.55, 'y250': 0.18, 'other': 0.05}),
        t,
      );
      expect(a.verdict, FrameVerdict.currency);
      expect(a.key, 'y500');
    });

    test('OOD GATE: a foreign note that lights up "other" is NOT announced as '
        'a Yemeni denomination', () {
      // e.g. the green 5-rupee field case: top Yemeni class competes with other.
      final FrameAssessment a = classifyFrame(
        e(<String, double>{'y500': 0.46, 'other': 0.40, 'y250': 0.30}),
        t,
      );
      expect(a.verdict, isNot(FrameVerdict.currency));
    });

    test('OOD CEILING: a confidently-WRONG denomination with a small-but-real '
        '"other" signal is refused (the Indian 5-rupee field case: y200_old 60% '
        '/ other 13%)', () {
      final FrameAssessment a = classifyFrame(
        e(<String, double>{'y200': 0.60, 'other': 0.13, 'y1000': 0.08}),
        t,
      );
      expect(a.verdict, isNot(FrameVerdict.currency));
    });

    test('REGRESSION (OOD via full distribution): a foreign note whose "other" '
        'mass sits at rank 4+ is still refused — the gate must see the FULL '
        'distribution, not a top-3 that drops "other"', () {
      // Realistic foreign-note softmax: two y200 sub-variants + y500 fill the
      // top-3, pushing the real "other" mass (0.14) to rank 4. If TFLiteService
      // truncated to the top-3 (the fixed bug), otherProb would arrive as 0 and
      // this would be announced as a confident "مائتا ريال". With the full
      // distribution carried through, otherProb=0.14 > otherCeiling(0.12) and
      // the note is correctly refused.
      final FrameAssessment a = classifyFrame(
        groupTopK(
          e(<String, double>{
            'y200_old': 0.32,
            'y200_new': 0.28,
            'y500': 0.15,
            'other': 0.14, // rank 4 — dropped by any top-3 truncation
            'y1000': 0.05,
            'y250': 0.03,
            'y50': 0.02,
            'y100_new': 0.005,
            'y100_old': 0.005,
          }),
        ),
        t,
      );
      expect(a.verdict, isNot(FrameVerdict.currency));
    });

    test('REGRESSION (grouping via full distribution): a worn y100 whose two '
        'designs are split by higher-scoring distractors still sums to a '
        'confident y100 — both sub-variants must be summed even when one ranks '
        '4th+', () {
      // y100_new (0.40) leads, but y500/y250 outrank the weak y100_old echo
      // (0.14), so y100_old is rank 4. Top-3 truncation would group y100 at only
      // 0.40 and reject a real bill as "uncertain". The full distribution sums
      // y100 to 0.54 and announces it.
      final FrameAssessment a = classifyFrame(
        groupTopK(
          e(<String, double>{
            'y100_new': 0.40,
            'y500': 0.16,
            'y250': 0.15,
            'y100_old': 0.14, // rank 4 — dropped by any top-3 truncation
            'other': 0.11,
            'y1000': 0.02,
            'y50': 0.01,
            'y200_new': 0.005,
            'y200_old': 0.005,
          }),
        ),
        t,
      );
      expect(a.verdict, FrameVerdict.currency);
      expect(a.key, 'y100');
    });

    test('decisively non-Yemeni: dominant "other" yields unknown', () {
      final FrameAssessment a = classifyFrame(
        e(<String, double>{'other': 0.62, 'y500': 0.20, 'y250': 0.10}),
        t,
      );
      expect(a.verdict, FrameVerdict.unknown);
      expect(a.key, 'other');
    });

    test('near-tie between two DIFFERENT denominations is not announced', () {
      final FrameAssessment a = classifyFrame(
        e(<String, double>{'y200': 0.40, 'y500': 0.38, 'other': 0.05}),
        t,
      );
      expect(a.verdict, FrameVerdict.uncertain);
    });

    test('below the confidence floor is uncertain', () {
      final FrameAssessment a = classifyFrame(
        e(<String, double>{'y50': 0.30, 'y100': 0.10, 'other': 0.08}),
        t,
      );
      expect(a.verdict, FrameVerdict.uncertain);
    });

    test('empty distribution is uncertain', () {
      expect(
        classifyFrame(<MapEntry<String, double>>[], t).verdict,
        FrameVerdict.uncertain,
      );
    });
  });

  group('MajorityVoter', () {
    final DateTime t0 = DateTime(2026, 6, 4, 12, 0, 0);

    test('requires the minimum match floor before any winner', () {
      final MajorityVoter v = MajorityVoter();
      v.add('y100', t0);
      v.add('y100', t0);
      expect(v.winner(t0), isNull); // only 2 votes, floor is 3
    });

    test('a steady denomination wins by strict majority', () {
      final MajorityVoter v = MajorityVoter();
      v.add('y100', t0);
      v.add('y100', t0);
      v.add('y100', t0);
      expect(v.winner(t0), 'y100');
    });

    test('a 3-of-10 plurality does NOT win (not a majority)', () {
      final MajorityVoter v = MajorityVoter();
      // 3 x y100 plus 7 other distinct labels → y100 is plurality but not >50%.
      v.add('y100', t0);
      v.add('y100', t0);
      v.add('y100', t0);
      for (int i = 0; i < 7; i++) {
        v.add('lbl$i', t0);
      }
      expect(v.winner(t0), isNull);
    });

    test('a 3-vs-3 split returns null (deterministic, never a coin-flip)', () {
      final MajorityVoter v = MajorityVoter();
      v.add('y100', t0);
      v.add('y200', t0);
      v.add('y100', t0);
      v.add('y200', t0);
      v.add('y100', t0);
      v.add('y200', t0);
      expect(v.winner(t0), isNull); // 3 vs 3, neither is >50%
    });

    test('votes older than the window age are pruned', () {
      final MajorityVoter v = MajorityVoter(maxAge: Duration(seconds: 5));
      v.add('y100', t0);
      v.add('y100', t0);
      v.add('y100', t0);
      // 10s later, all three are stale → pruned → no winner.
      final DateTime later = t0.add(const Duration(seconds: 10));
      expect(v.winner(later), isNull);
    });

    test('clear() empties the window', () {
      final MajorityVoter v = MajorityVoter();
      v.add('y100', t0);
      v.add('y100', t0);
      v.add('y100', t0);
      v.clear();
      expect(v.length, 0);
      expect(v.winner(t0), isNull);
    });
  });
}
