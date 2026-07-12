import 'dart:async';

import 'package:bigash/data/datasources/camera_service.dart';
import 'package:bigash/data/models/currency_model.dart';
import 'package:bigash/data/services/tflite_service.dart';
import 'package:bigash/domain/currency/yemeni_currency_label_mapper.dart';
import 'package:bigash/domain/entities/currency_entity.dart';
import 'package:bigash/domain/repositories/currency_repository.dart';
import 'package:camera/camera.dart';

class CurrencyRepositoryImpl implements CurrencyRepository {
  final CameraService cameraService;
  final TFLiteService tfliteService;

  StreamSubscription<CameraFrame>? _cameraSubscription;

  CurrencyRepositoryImpl({
    required this.cameraService,
    required this.tfliteService,
  });

  @override
  bool get isModelLoaded => tfliteService.isModelLoaded;

  @override
  bool get isModelReady => tfliteService.isModelReady;

  @override
  String? get modelErrorMessage => tfliteService.contractError;

  @override
  bool get isCameraReady =>
      cameraService.isInitialized && cameraService.isStreaming;

  @override
  CameraController? get cameraController => cameraService.cameraController;

  @override
  Future<void> initialize() async {
    await tfliteService.initialize();
  }

  @override
  Future<void> startCamera() async {
    await _cameraSubscription?.cancel();
    _cameraSubscription = null;
    await cameraService.initialize();
    if (!cameraService.isStreaming) {
      return;
    }
    _cameraSubscription = cameraService.frameStream.listen((frame) {
      tfliteService.processImage(
        frame.image,
        sensorOrientation: frame.sensorOrientation,
        mirror: frame.isFrontFacing,
      );
    });
  }

  @override
  Future<void> stopCamera() async {
    await _cameraSubscription?.cancel();
    _cameraSubscription = null;
    await cameraService.setTorchEnabled(false);
    await cameraService.disposeCamera();
  }

  @override
  Future<void> pauseCamera() async {
    // Keep the subscription alive — we'll resume from where we paused.
    await cameraService.setTorchEnabled(false);
    await cameraService.pauseStreaming();
  }

  @override
  Future<bool> resumeCamera() => cameraService.resumeStreaming();

  @override
  Future<bool> setTorchEnabled(bool enabled) =>
      cameraService.setTorchEnabled(enabled);

  @override
  Stream<CurrencyEntity> getCurrencyStream() {
    return tfliteService.resultStream.map((result) {
      final String normalized = normalizeYemeniClassifierLabel(result.label);
      return CurrencyModel.fromInference(
        label: normalized,
        displayName: arabicTtsForYemeniBill(normalized),
        confidence: result.confidence,
        topK: result.topK,
        quality: result.quality,
        brightness: result.brightness,
        sharpness: result.sharpness,
      );
    });
  }

  @override
  void dispose() {
    unawaited(_cameraSubscription?.cancel());
    _cameraSubscription = null;
    cameraService.dispose();
    tfliteService.dispose();
  }
}
