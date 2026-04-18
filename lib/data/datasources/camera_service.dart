import 'dart:async';
import 'dart:io';
import 'package:camera/camera.dart';

class CameraFrame {
  final CameraImage image;
  final int sensorOrientation;
  final bool isFrontFacing;

  const CameraFrame({
    required this.image,
    required this.sensorOrientation,
    required this.isFrontFacing,
  });
}

class CameraService {
  CameraController? _cameraController;
  CameraDescription? _selectedCamera;

  final StreamController<CameraFrame> _frameStreamController =
      StreamController<CameraFrame>.broadcast();

  Stream<CameraFrame> get frameStream => _frameStreamController.stream;
  CameraController? get cameraController => _cameraController;
  int get sensorOrientation => _selectedCamera?.sensorOrientation ?? 90;
  bool get isFrontFacing =>
      _selectedCamera?.lensDirection == CameraLensDirection.front;

  Future<void> initialize() async {
    if (_cameraController?.value.isInitialized == true) {
      return;
    }

    final cameras = await availableCameras();
    if (cameras.isEmpty) {
      return;
    }

    // Prefer the back camera for reading bills held in hand.
    _selectedCamera = cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.back,
      orElse: () => cameras.first,
    );

    _cameraController = CameraController(
      _selectedCamera!,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: Platform.isAndroid
          ? ImageFormatGroup.yuv420
          : ImageFormatGroup.bgra8888,
    );

    await _cameraController!.initialize();
    await _cameraController!.setFocusMode(FocusMode.auto).catchError((_) {});
    await _cameraController!.setExposureMode(ExposureMode.auto).catchError((_) {});

    final int orientation = _selectedCamera!.sensorOrientation;
    final bool front = _selectedCamera!.lensDirection == CameraLensDirection.front;

    _cameraController!.startImageStream((CameraImage image) {
      if (!_frameStreamController.isClosed) {
        _frameStreamController.add(
          CameraFrame(
            image: image,
            sensorOrientation: orientation,
            isFrontFacing: front,
          ),
        );
      }
    });
  }

  Future<void> setTorchEnabled(bool enabled) async {
    final CameraController? controller = _cameraController;
    if (controller == null || !controller.value.isInitialized) {
      return;
    }
    try {
      await controller.setFlashMode(
        enabled ? FlashMode.torch : FlashMode.off,
      );
    } catch (_) {
      // Torch may be unavailable on some devices.
    }
  }

  Future<void> disposeCamera() async {
    final CameraController? controller = _cameraController;
    if (controller == null) {
      return;
    }

    if (controller.value.isStreamingImages) {
      await controller.stopImageStream();
    }
    await controller.dispose();
    _cameraController = null;
    _selectedCamera = null;
  }

  void dispose() {
    unawaited(disposeCamera());
    _frameStreamController.close();
  }
}
