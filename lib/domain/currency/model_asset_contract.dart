import 'package:bigash/domain/currency/yemeni_currency_label_mapper.dart';

/// SHA-256 of the exact safety-reviewed model shipped as bigash_v18.tflite.
/// A retrained model must be evaluated, copied into assets, and this digest
/// updated deliberately in the same change.
const String expectedBigashModelSha256 =
    '7822eca365385d1ee86052f5ada1c226fca45d611aeff9af61def33263a2e2b2';

const List<String> expectedBigashLabels = <String>[
  'other',
  'y1000',
  'y100_new',
  'y100_old',
  'y200_new',
  'y200_old',
  'y250',
  'y50',
  'y500',
];

bool labelsMatchExpectedBigashOrder(Iterable<String> rawLabels) {
  final List<String> normalized = rawLabels
      .map(normalizeYemeniClassifierLabel)
      .where((String label) => label.isNotEmpty)
      .toList(growable: false);
  if (normalized.length != expectedBigashLabels.length) return false;
  for (int i = 0; i < normalized.length; i++) {
    if (normalized[i] != expectedBigashLabels[i]) return false;
  }
  return true;
}
