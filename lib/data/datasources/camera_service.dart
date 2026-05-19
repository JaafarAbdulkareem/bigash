import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';

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

  /// Tracks whether [initialize] succeeded. Surfaces useful diagnostics
  /// for the splash / scanner UI when something silently fails.
  String? _lastInitError;
  ResolutionPreset? _activeResolution;

  String? get lastInitError => _lastInitError;
  ResolutionPreset? get activeResolution => _activeResolution;

  final StreamController<CameraFrame> _frameStreamController =
      StreamController<CameraFrame>.broadcast();

  Stream<CameraFrame> get frameStream => _frameStreamController.stream;
  CameraController? get cameraController => _cameraController;
  int get sensorOrientation => _selectedCamera?.sensorOrientation ?? 90;
  bool get isFrontFacing =>
      _selectedCamera?.lensDirection == CameraLensDirection.front;

  bool get isInitialized => _cameraController?.value.isInitialized ?? false;

  Future<void> initialize() async {
    if (_cameraController?.value.isInitialized == true) {
      return;
    }

    _lastInitError = null;

    final List<CameraDescription> cameras;
    try {
      cameras = await availableCameras();
    } catch (e) {
      _lastInitError = 'تعذر الوصول إلى الكاميرا: $e';
      debugPrint('[Camera] availableCameras failed: $e');
      return;
    }

    if (cameras.isEmpty) {
      _lastInitError = 'لم يُعثر على أي كاميرا في الجهاز';
      return;
    }

    // Prefer the back camera for reading bills held in hand.
    _selectedCamera = cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.back,
      orElse: () => cameras.first,
    );

    // Try the highest resolution we can. If that fails (rare, but happens on
    // budget devices), fall back to lower presets. Better detail → better
    // currency recognition, especially for small denominations.
    final List<ResolutionPreset> presets = <ResolutionPreset>[
      ResolutionPreset.high, // 1280x720 — sweet spot for bills
      ResolutionPreset.medium, // 640x480 — fallback
      ResolutionPreset.low, // 320x240 — last resort
    ];

    bool initialized = false;
    for (final ResolutionPreset preset in presets) {
      _cameraController = CameraController(
        _selectedCamera!,
        preset,
        enableAudio: false,
        imageFormatGroup: Platform.isAndroid
            ? ImageFormatGroup.yuv420
            : ImageFormatGroup.bgra8888,
      );

      try {
        await _cameraController!.initialize();
        _activeResolution = preset;
        debugPrint('[Camera] initialized at $preset');
        initialized = true;
        break;
      } catch (e) {
        debugPrint('[Camera] preset=$preset failed: $e');
        try {
          await _cameraController!.dispose();
        } catch (_) {}
        _cameraController = null;
      }
    }

    if (!initialized) {
      _lastInitError = 'فشل تهيئة الكاميرا بجميع مستويات الدقة';
      return;
    }

    // Best-effort focus and exposure setup. Some devices (notably older
    // Androids and some emulators) don't support these — failures are ignored.
    try {
      await _cameraController!.setFocusMode(FocusMode.auto);
    } catch (e) {
      debugPrint('[Camera] setFocusMode failed: $e');
    }
    try {
      await _cameraController!.setExposureMode(ExposureMode.auto);
    } catch (e) {
      debugPrint('[Camera] setExposureMode failed: $e');
    }

    final int orientation = _selectedCamera!.sensorOrientation;
    final bool front =
        _selectedCamera!.lensDirection == CameraLensDirection.front;

    try {
      await _cameraController!.startImageStream((CameraImage image) {
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
    } catch (e) {
      _lastInitError = 'فشل تشغيل بث الإطارات من الكاميرا: $e';
      debugPrint('[Camera] startImageStream failed: $e');
    }
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

  /// Pauses the stream without disposing the controller — useful when the
  /// app goes into the background. Resume with [resumeStreaming].
  Future<void> pauseStreaming() async {
    final CameraController? controller = _cameraController;
    if (controller == null || !controller.value.isInitialized) {
      return;
    }
    if (controller.value.isStreamingImages) {
      try {
        await controller.stopImageStream();
      } catch (e) {
        debugPrint('[Camera] pauseStreaming failed: $e');
      }
    }
  }

  Future<void> resumeStreaming() async {
    final CameraController? controller = _cameraController;
    if (controller == null || !controller.value.isInitialized) {
      return;
    }
    if (controller.value.isStreamingImages) {
      return;
    }
    final int orientation = sensorOrientation;
    final bool front = isFrontFacing;
    try {
      await controller.startImageStream((CameraImage image) {
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
    } catch (e) {
      debugPrint('[Camera] resumeStreaming failed: $e');
    }
  }

  Future<void> disposeCamera() async {
    final CameraController? controller = _cameraController;
    if (controller == null) {
      return;
    }

    if (controller.value.isStreamingImages) {
      try {
        await controller.stopImageStream();
      } catch (_) {}
    }
    try {
      await controller.dispose();
    } catch (_) {}
    _cameraController = null;
    _selectedCamera = null;
    _activeResolution = null;
  }

  void dispose() {
    unawaited(disposeCamera());
    _frameStreamController.close();
  }
}
