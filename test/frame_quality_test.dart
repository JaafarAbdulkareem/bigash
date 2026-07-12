import 'package:bigash/domain/currency/frame_quality.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  const FrameQualityMetrics usable = FrameQualityMetrics(
    meanLuma: 120,
    meanGradient: 12,
    darkRatio: 0.05,
    brightRatio: 0.04,
  );

  test('accepts a normally exposed sharp frame', () {
    expect(assessFrameQuality(usable), CaptureQuality.usable);
  });

  test('rejects motion-blurred or textureless frames', () {
    expect(
      assessFrameQuality(
        const FrameQualityMetrics(
          meanLuma: 120,
          meanGradient: 2.5,
          darkRatio: 0.05,
          brightRatio: 0.05,
        ),
      ),
      CaptureQuality.blurry,
    );
  });

  test('rejects underexposed frames before blur is considered', () {
    expect(
      assessFrameQuality(
        const FrameQualityMetrics(
          meanLuma: 25,
          meanGradient: 1,
          darkRatio: 0.85,
          brightRatio: 0,
        ),
      ),
      CaptureQuality.tooDark,
    );
  });

  test('rejects overexposed frames', () {
    expect(
      assessFrameQuality(
        const FrameQualityMetrics(
          meanLuma: 245,
          meanGradient: 8,
          darkRatio: 0,
          brightRatio: 0.9,
        ),
      ),
      CaptureQuality.tooBright,
    );
  });

  test('rejects non-finite metrics defensively', () {
    expect(
      assessFrameQuality(
        const FrameQualityMetrics(
          meanLuma: double.nan,
          meanGradient: 8,
          darkRatio: 0,
          brightRatio: 0,
        ),
      ),
      CaptureQuality.blurry,
    );
  });
}
