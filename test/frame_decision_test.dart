// Unit tests for the safety-critical decision core: grouping the classifier's
// raw sub-variant probabilities, the per-frame confidence/margin/OOD gates, and
// the time-bounded majority voter. These are the pure functions that decide
// what denomination (if any) gets spoken — so they are tested directly, with no
// camera/TTS/vibration plugins in the loop.

import 'package:flutter_test/flutter_test.dart';
import 'package:bigash/domain/currency/frame_decision.dart';

List<MapEntry<String, double>> e(Map<String, double> m) =>
    m.entries.toList();

void main() {
  group('groupTopK', () {
    test('sums the new/old sub-variants of one denomination into the parent', () {
      // A worn 100 bill splits its mass across its two designs.
      final List<MapEntry<String, double>> grouped = groupTopK(<MapEntry<String, double>>[
        const MapEntry<String, double>('y100_new', 0.31),
        const MapEntry<String, double>('y100_old', 0.27),
        const MapEntry<String, double>('other', 0.09),
      ]);
      expect(grouped.first.key, 'y100');
      expect(grouped.first.value, closeTo(0.58, 1e-9));
      expect(grouped[1].key, 'other');
    });

    test('strips index prefixes and is deterministic on ties', () {
      final List<MapEntry<String, double>> grouped = groupTopK(<MapEntry<String, double>>[
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
      expect(classifyFrame(<MapEntry<String, double>>[], t).verdict,
          FrameVerdict.uncertain);
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
