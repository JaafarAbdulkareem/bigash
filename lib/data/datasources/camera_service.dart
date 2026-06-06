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

    // The model only consumes a 224x224 crop, so medium (640x480) already
    // carries ample detail while doing ~4x less per-frame YUV->RGB
    // conversion work than 720p — which keeps the scan responsive on
    // mid-range phones. Fall back to low only if medium is unsupported.
    final List<ResolutionPreset> presets = <ResolutionPreset>[
      ResolutionPreset.medium, // 640x480 — preferred
      ResolutionPreset.low, // 320x240 — fallback
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

  /// Returns true if the flash mode was actually changed, false if there is no
  /// usable controller or the device rejected the request — so the caller can
  /// keep its cached torch state in sync with the hardware.
  Future<bool> setTorchEnabled(bool enabled) async {
    final CameraController? controller = _cameraController;
    if (controller == null || !controller.value.isInitialized) {
      return false;
    }
    try {
      await controller.setFlashMode(
        enabled ? FlashMode.torch : FlashMode.off,
      );
      return true;
    } catch (_) {
      // Torch may be unavailable on some devices.
      return false;
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

  /// Returns true only if the camera is actually streaming frames again after
  /// the call. On backgrounding (especially iOS) the capture session can be
  /// torn down/invalidated; returning the real streaming state lets the caller
  /// avoid leaving a frozen "scanning" screen with no live frames.
  Future<bool> resumeStreaming() async {
    final CameraController? controller = _cameraController;
    if (controller == null || !controller.value.isInitialized) {
      return false;
    }
    if (controller.value.isStreamingImages) {
      return true;
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
      return controller.value.isStreamingImages;
    } catch (e) {
      debugPrint('[Camera] resumeStreaming failed: $e');
      return false;
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
