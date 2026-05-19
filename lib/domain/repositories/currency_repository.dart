import 'package:bigash/domain/entities/currency_entity.dart';
import 'package:camera/camera.dart';

abstract class CurrencyRepository {
  bool get isModelLoaded;
  String get diagnostics;

  Stream<CurrencyEntity> getCurrencyStream();
  Future<void> initialize();
  Future<void> startCamera();
  Future<void> stopCamera();
  /// Pauses camera streaming without disposing — for app backgrounding.
  Future<void> pauseCamera();
  /// Resumes camera streaming after a [pauseCamera] call.
  Future<void> resumeCamera();
  Future<void> setTorchEnabled(bool enabled);
  void cycleNormalization();
  CameraController? get cameraController;
  void dispose();
}
