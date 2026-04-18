/// Normalizes classifier labels such as `6 y1000`, `6_y1000`, or `y500` to a canonical key (`y1000`, `y500`).
String normalizeYemeniClassifierLabel(String raw) {
  var s = raw.trim().toLowerCase();
  s = s.replaceFirst(RegExp(r'^\d+[\s_]+'), '');
  return s.trim();
}

bool isIgnoredYemeniCurrencyLabel(String normalized) {
  return normalized == 'other' ||
      normalized == 'background' ||
      normalized == '0_background' ||
      normalized == '1_other';
}

/// Clear Arabic for TTS (Yemeni riyal denominations).
String arabicTtsForYemeniBill(String normalized) {
  switch (normalized) {
    case 'y50':
      return 'خمسون ريال يمني';
    case 'y100':
      return 'مائة ريال يمني';
    case 'y200':
      return 'مائتان ريال يمني';
    case 'y250':
      return 'مائتان وخمسون ريال يمني';
    case 'y500':
      return 'خمسمائة ريال يمني';
    case 'y1000':
      return 'ألف ريال يمني';
    default:
      return normalized;
  }
}
