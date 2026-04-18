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

  StreamSubscription? _cameraSubscription;

  CurrencyRepositoryImpl({
    required this.cameraService,
    required this.tfliteService,
  });

  @override
  bool get isModelLoaded => tfliteService.isModelLoaded;

  @override
  CameraController? get cameraController => cameraService.cameraController;

  @override
  Future<void> initialize() async {
    await tfliteService.initialize();
  }

  @override
  Future<void> startCamera() async {
    await cameraService.initialize();
    _cameraSubscription?.cancel();
    _cameraSubscription = cameraService.imageStream.listen((image) {
      tfliteService.processImage(image);
    });
  }

  @override
  Future<void> stopCamera() async {
    _cameraSubscription?.cancel();
    await cameraService.setTorchEnabled(false);
    await cameraService.disposeCamera();
  }

  @override
  Future<void> setTorchEnabled(bool enabled) =>
      cameraService.setTorchEnabled(enabled);

  @override
  Stream<CurrencyEntity> getCurrencyStream() {
    return tfliteService.resultStream.map((result) {
      final String normalized =
          normalizeYemeniClassifierLabel(result.label);
      return CurrencyModel.fromInference(
        label: normalized,
        displayName: arabicTtsForYemeniBill(normalized),
        confidence: result.confidence,
      );
    });
  }

  @override
  void dispose() {
    _cameraSubscription?.cancel();
    cameraService.dispose();
    tfliteService.dispose();
  }
}
