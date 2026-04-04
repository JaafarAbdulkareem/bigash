import 'package:bigash/domain/entities/currency_entity.dart';
import 'package:camera/camera.dart';

abstract class CurrencyRepository {
  bool get isModelLoaded;

  Stream<CurrencyEntity> getCurrencyStream();
  Future<void> initialize();
  Future<void> startCamera();
  Future<void> stopCamera();
  CameraController? get cameraController;
  void dispose();
}
