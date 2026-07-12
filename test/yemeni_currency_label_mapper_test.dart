// Unit tests for the Yemeni-currency label mapper — the pure logic that
// turns raw classifier labels into canonical keys and Arabic speech.

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/services.dart';
import 'package:bigash/domain/currency/yemeni_currency_label_mapper.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('normalizeYemeniClassifierLabel', () {
    test('strips a leading numeric index separated by a space', () {
      expect(normalizeYemeniClassifierLabel('6 y1000'), 'y1000');
      expect(normalizeYemeniClassifierLabel('0 other'), 'other');
    });

    test('strips a leading numeric index separated by an underscore', () {
      expect(normalizeYemeniClassifierLabel('6_y1000'), 'y1000');
    });

    test('lowercases and trims surrounding whitespace', () {
      expect(normalizeYemeniClassifierLabel('  Y500  '), 'y500');
    });

    test('passes an already-clean label through unchanged', () {
      expect(normalizeYemeniClassifierLabel('y50'), 'y50');
    });
  });

  group('isIgnoredYemeniCurrencyLabel', () {
    test('treats non-currency labels as ignored', () {
      expect(isIgnoredYemeniCurrencyLabel('other'), isTrue);
      expect(isIgnoredYemeniCurrencyLabel('background'), isTrue);
    });

    test('treats real denominations as not ignored', () {
      for (final String label in <String>[
        'y50',
        'y100',
        'y200',
        'y250',
        'y500',
        'y1000',
      ]) {
        expect(isIgnoredYemeniCurrencyLabel(label), isFalse, reason: label);
      }
    });
  });

  group('arabicTtsForYemeniBill', () {
    test('maps every supported denomination to Arabic speech', () {
      expect(arabicTtsForYemeniBill('y50'), 'خمسون ريال يمني');
      expect(arabicTtsForYemeniBill('y100'), 'مائة ريال يمني');
      // "مائتا" not "مائتان": nūn of the dual drops in iḍāfa before the noun.
      expect(arabicTtsForYemeniBill('y200'), 'مائتا ريال يمني');
      expect(arabicTtsForYemeniBill('y250'), 'مائتان وخمسون ريال يمني');
      expect(arabicTtsForYemeniBill('y500'), 'خمسمائة ريال يمني');
      expect(arabicTtsForYemeniBill('y1000'), 'ألف ريال يمني');
    });

    test('new/old sub-classes share the spoken phrase of the denomination', () {
      // The user cannot distinguish bill design — only the value matters.
      expect(arabicTtsForYemeniBill('y100_new'), 'مائة ريال يمني');
      expect(arabicTtsForYemeniBill('y100_old'), 'مائة ريال يمني');
      expect(arabicTtsForYemeniBill('y200_new'), 'مائتا ريال يمني');
      expect(arabicTtsForYemeniBill('y200_old'), 'مائتا ريال يمني');
    });

    test(
      'every label actually shipped in assets/labels.txt maps to non-empty '
      'Arabic speech (no raw English token spoken to a blind user)',
      () async {
        // Safety contract: the 9 classes the model can actually emit must each
        // resolve to a real Arabic phrase, never fall through to a raw token.
        final String content = await rootBundle.loadString('assets/labels.txt');
        final List<String> shipped = content
            .split('\n')
            .map(normalizeYemeniClassifierLabel)
            .where((String label) => label.isNotEmpty)
            .toList();
        expect(shipped, hasLength(9));
        for (final String label in shipped) {
          if (isIgnoredYemeniCurrencyLabel(label)) continue;
          final String spoken = arabicTtsForYemeniBill(label);
          expect(
            spoken,
            isNot(label),
            reason: '$label fell through to a verbatim token',
          );
          expect(spoken.contains('ريال'), isTrue, reason: label);
        }
      },
    );

    test('falls back to the raw label for unknown input', () {
      expect(arabicTtsForYemeniBill('xyz'), 'xyz');
    });
  });

  test('end to end: raw model label maps to Arabic speech', () {
    final String key = normalizeYemeniClassifierLabel('2 y1000');
    expect(key, 'y1000');
    expect(isIgnoredYemeniCurrencyLabel(key), isFalse);
    expect(arabicTtsForYemeniBill(key), 'ألف ريال يمني');
  });

  group('denominationGroupForVoting', () {
    test(
      'collapses 100 and 200 design variants to the parent denomination',
      () {
        expect(denominationGroupForVoting('y100_new'), 'y100');
        expect(denominationGroupForVoting('y100_old'), 'y100');
        expect(denominationGroupForVoting('y200_new'), 'y200');
        expect(denominationGroupForVoting('y200_old'), 'y200');
      },
    );

    test('passes single-design denominations through unchanged', () {
      for (final String label in <String>['y50', 'y250', 'y500', 'y1000']) {
        expect(denominationGroupForVoting(label), label, reason: label);
      }
    });

    test('passes non-currency labels through unchanged', () {
      expect(denominationGroupForVoting('other'), 'other');
    });
  });
}
