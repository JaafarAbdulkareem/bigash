import 'dart:async';
import 'dart:io';
import 'package:camera/camera.dart';

class CameraService {
  CameraController? _cameraController;
  final StreamController<CameraImage> _imageStreamController =
      StreamController<CameraImage>.broadcast();

  Stream<CameraImage> get imageStream => _imageStreamController.stream;
  CameraController? get cameraController => _cameraController;

  Future<void> initialize() async {
    if (_cameraController?.value.isInitialized == true) {
      return;
    }

    final cameras = await availableCameras();
    if (cameras.isEmpty) {
      return;
    }

    _cameraController = CameraController(
      cameras[0],
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup:
          Platform.isAndroid ? ImageFormatGroup.yuv420 : ImageFormatGroup.bgra8888,
    );

    await _cameraController!.initialize();

    _cameraController!.startImageStream((CameraImage image) {
      if (!_imageStreamController.isClosed) {
        _imageStreamController.add(image);
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
  }

  void dispose() {
    unawaited(disposeCamera());
    _imageStreamController.close();
  }
}
