// Unit tests for the Yemeni-currency label mapper — the pure logic that
// turns raw classifier labels into canonical keys and Arabic speech.

import 'package:flutter_test/flutter_test.dart';
import 'package:bigash/domain/currency/yemeni_currency_label_mapper.dart';

void main() {
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
      for (final String label in <String>['y50', 'y100', 'y200', 'y250',
        'y500', 'y1000']) {
        expect(isIgnoredYemeniCurrencyLabel(label), isFalse, reason: label);
      }
    });
  });

  group('arabicTtsForYemeniBill', () {
    test('maps every supported denomination to Arabic speech', () {
      expect(arabicTtsForYemeniBill('y50'), 'خمسون ريال يمني');
      expect(arabicTtsForYemeniBill('y100'), 'مائة ريال يمني');
      expect(arabicTtsForYemeniBill('y200'), 'مائتان ريال يمني');
      expect(arabicTtsForYemeniBill('y250'), 'مائتان وخمسون ريال يمني');
      expect(arabicTtsForYemeniBill('y500'), 'خمسمائة ريال يمني');
      expect(arabicTtsForYemeniBill('y1000'), 'ألف ريال يمني');
    });

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
}
