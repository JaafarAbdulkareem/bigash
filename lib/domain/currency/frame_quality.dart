/// Quality of the camera crop before it is allowed into the classifier.
enum CaptureQuality { usable, blurry, tooDark, tooBright }

class FrameQualityMetrics {
  const FrameQualityMetrics({
    required this.meanLuma,
    required this.meanGradient,
    required this.darkRatio,
    required this.brightRatio,
  });

  /// Average perceived brightness on the 0..255 scale.
  final double meanLuma;

  /// Average local luminance change. Very low values mean that the crop has
  /// little usable edge detail, usually because of motion blur, bad focus, or
  /// a banknote that is too far from the camera.
  final double meanGradient;

  final double darkRatio;
  final double brightRatio;
}

class FrameQualityThresholds {
  const FrameQualityThresholds({
    this.minimumLuma = 35,
    this.maximumLuma = 235,
    this.minimumGradient = 3.5,
    this.maximumDarkRatio = 0.78,
    this.maximumBrightRatio = 0.82,
  });

  final double minimumLuma;
  final double maximumLuma;
  final double minimumGradient;
  final double maximumDarkRatio;
  final double maximumBrightRatio;
}

/// Safety-first image gate: unusable frames are never classified or voted.
CaptureQuality assessFrameQuality(
  FrameQualityMetrics metrics, {
  FrameQualityThresholds thresholds = const FrameQualityThresholds(),
}) {
  if (!metrics.meanLuma.isFinite ||
      !metrics.meanGradient.isFinite ||
      !metrics.darkRatio.isFinite ||
      !metrics.brightRatio.isFinite) {
    return CaptureQuality.blurry;
  }
  if (metrics.meanLuma < thresholds.minimumLuma ||
      metrics.darkRatio > thresholds.maximumDarkRatio) {
    return CaptureQuality.tooDark;
  }
  if (metrics.meanLuma > thresholds.maximumLuma ||
      metrics.brightRatio > thresholds.maximumBrightRatio) {
    return CaptureQuality.tooBright;
  }
  if (metrics.meanGradient < thresholds.minimumGradient) {
    return CaptureQuality.blurry;
  }
  return CaptureQuality.usable;
}
