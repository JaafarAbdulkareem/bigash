import 'package:bigash/domain/entities/currency_entity.dart';

/// Sub-state of the strict verification guard (no partial / chatty speech).
enum DetectionVerificationState {
  stableMatch,
  inconsistent,
  lowConfidence,
}

/// One throttled inference sample pushed into the 5-frame buffer.
class DetectionFrame {
  final String label;
  final String displayName;
  final double confidence;

  const DetectionFrame({
    required this.label,
    required this.displayName,
    required this.confidence,
  });
}

/// Result of applying one frame to the guard (pure domain logic).
class StrictVerificationResult {
  final List<DetectionFrame> newBuffer;
  final DetectionVerificationState verificationState;
  final String? statusHint;
  final String? speakText;
  final bool shouldVibrate;

  const StrictVerificationResult({
    required this.newBuffer,
    required this.verificationState,
    this.statusHint,
    this.speakText,
    this.shouldVibrate = false,
  });
}

class RecognizeCurrencyUseCase {
  static const double confidenceThreshold = 0.85;
  static const int bufferSize = 5;

  static const Set<String> ignoredLabels = {'0_background', '1_other'};

  /// Strict rule: speak only if the last [bufferSize] frames share the same label
  /// and every confidence is **strictly** above [confidenceThreshold].
  StrictVerificationResult applyFrame({
    required List<DetectionFrame> buffer,
    required CurrencyEntity entity,
  }) {
    if (ignoredLabels.contains(entity.label)) {
      return StrictVerificationResult(
        newBuffer: <DetectionFrame>[],
        verificationState: DetectionVerificationState.inconsistent,
        statusHint: 'جاري المسح',
        shouldVibrate: true,
      );
    }

    if (entity.confidence <= confidenceThreshold) {
      return StrictVerificationResult(
        newBuffer: <DetectionFrame>[],
        verificationState: DetectionVerificationState.lowConfidence,
        statusHint: 'لم يُثبت الإطار بعد',
        shouldVibrate: true,
      );
    }

    final List<DetectionFrame> next = List<DetectionFrame>.from(buffer)
      ..add(
        DetectionFrame(
          label: entity.label,
          displayName: entity.displayName,
          confidence: entity.confidence,
        ),
      );

    if (next.length > bufferSize) {
      next.removeRange(0, next.length - bufferSize);
    }

    if (next.length < bufferSize) {
      return StrictVerificationResult(
        newBuffer: next,
        verificationState: DetectionVerificationState.inconsistent,
        statusHint: 'جاري التحقق من الاستقرار',
        shouldVibrate: false,
      );
    }

    final String canonical = next.first.label;
    final bool allSameLabel = next.every((DetectionFrame e) => e.label == canonical);
    final bool allAboveThreshold =
        next.every((DetectionFrame e) => e.confidence > confidenceThreshold);

    if (allSameLabel && allAboveThreshold) {
      final String say = next.last.displayName;
      return StrictVerificationResult(
        newBuffer: <DetectionFrame>[],
        verificationState: DetectionVerificationState.stableMatch,
        statusHint: say,
        speakText: say,
        shouldVibrate: false,
      );
    }

    return StrictVerificationResult(
      newBuffer: <DetectionFrame>[],
      verificationState: DetectionVerificationState.inconsistent,
      statusHint: 'لم يتطابق الإطار مع الخمسة السابقة',
      shouldVibrate: true,
    );
  }
}
