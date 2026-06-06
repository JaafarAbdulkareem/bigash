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

/// Collapses `_new` / `_old` design variants into the parent denomination,
/// so the rolling-window vote in DetectionCubit counts both variants together.
///
/// Why: a worn 100-riyal bill may flip between `y100_new` and `y100_old`
/// across frames. Each variant maps to the same spoken phrase, so for the
/// user it is the same announcement — but if we vote on the raw label, the
/// frames split between two buckets and neither reaches the match threshold,
/// silently swallowing a correct detection.
String denominationGroupForVoting(String normalized) {
  switch (normalized) {
    case 'y100_new':
    case 'y100_old':
      return 'y100';
    case 'y200_new':
    case 'y200_old':
      return 'y200';
    default:
      return normalized;
  }
}

/// Clear Arabic for TTS (Yemeni riyal denominations). Plain, unvoweled text:
/// adding tashkeel marks made device TTS engines read it incorrectly.
///
/// y100/y200 are split into `_new` and `_old` sub-classes at the model level
/// (the new and old banknote designs are visually distinct enough that one
/// shared class let the more-populated old design dominate training). Both
/// sub-classes map to the same spoken phrase — the user only cares about
/// the denomination, not the design generation.
String arabicTtsForYemeniBill(String normalized) {
  switch (normalized) {
    case 'y50':
      return 'خمسون ريال يمني';
    case 'y100':
    case 'y100_new':
    case 'y100_old':
      return 'مائة ريال يمني';
    case 'y200':
    case 'y200_new':
    case 'y200_old':
      // "مائتا" (not "مائتان"): in iḍāfa before the counted noun the nūn of
      // the dual drops — "مِائَتَا رِيَالٍ". (y250 keeps "مائتان" because it is
      // followed by the conjunction "و", not by the counted noun directly.)
      return 'مائتا ريال يمني';
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
