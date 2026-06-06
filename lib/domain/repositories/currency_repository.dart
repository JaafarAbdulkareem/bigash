import 'package:bigash/domain/entities/currency_entity.dart';
import 'package:camera/camera.dart';

abstract class CurrencyRepository {
  bool get isModelLoaded;

  /// True only when the model is loaded AND its output class count matches the
  /// bundled labels (the contract). False if the model is missing OR the
  /// model/labels disagree — in which case [modelErrorMessage] explains why.
  bool get isModelReady;

  /// A spoken-ready Arabic explanation when [isModelReady] is false for a
  /// reason other than "no model file" (e.g. class-count/labels mismatch),
  /// or null when the generic "model not bundled" message should be used.
  String? get modelErrorMessage;

  /// True once the camera has successfully initialized and is streaming.
  bool get isCameraReady;

  Stream<CurrencyEntity> getCurrencyStream();
  Future<void> initialize();
  Future<void> startCamera();
  Future<void> stopCamera();
  /// Pauses camera streaming without disposing — for app backgrounding.
  Future<void> pauseCamera();
  /// Resumes camera streaming after a [pauseCamera] call. Returns true only if
  /// the camera is actually streaming frames again, so the caller can surface
  /// a failure instead of leaving a frozen "scanning" screen.
  Future<bool> resumeCamera();
  /// Returns true if the torch reached the requested state, false otherwise
  /// (e.g. no torch hardware / controller gone) so callers can keep their
  /// cached torch state truthful.
  Future<bool> setTorchEnabled(bool enabled);
  CameraController? get cameraController;
  void dispose();
}
