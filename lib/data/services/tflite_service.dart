import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class TFLiteResult {
  final String label;
  final String displayName;
  final double confidence;

  const TFLiteResult({
    required this.label,
    required this.displayName,
    required this.confidence,
  });
}

class TFLiteService {
  static const String modelAssetPath = 'assets/bigash_v15_final.tflite';

  /// Target ~6–7 FPS inference (between 5–10 Hz) to avoid chatty ML output.
  static const Duration _minFrameInterval = Duration(milliseconds: 150);

  Interpreter? _interpreter;
  List<String> _labels = <String>[];
  bool _isProcessing = false;
  DateTime? _lastEmittedAt;

  final StreamController<TFLiteResult> _streamController =
      StreamController<TFLiteResult>.broadcast();

  Stream<TFLiteResult> get resultStream => _streamController.stream;

  bool get isModelLoaded => _interpreter != null;

  Future<void> initialize() async {
    if (_interpreter == null) {
      try {
        _interpreter = await Interpreter.fromAsset(modelAssetPath);
      } on FlutterError catch (e) {
        debugPrint('TFLite model not bundled or load failed: $e');
      } catch (e) {
        debugPrint('TFLite model load failed: $e');
      }
    }

    try {
      final String labelsContent = await rootBundle.loadString('assets/labels.txt');
      _labels = labelsContent
          .split('\n')
          .map((String e) => e.trim())
          .where((String e) => e.isNotEmpty)
          .toList();
    } catch (e) {
      debugPrint('Labels load failed: $e');
      _labels = <String>[];
    }
  }

  Future<void> processImage(CameraImage image) async {
    if (_isProcessing || _interpreter == null || _labels.isEmpty) {
      return;
    }

    final DateTime now = DateTime.now();
    if (_lastEmittedAt != null &&
        now.difference(_lastEmittedAt!) < _minFrameInterval) {
      return;
    }

    _isProcessing = true;
    try {
      final img.Image? converted = _convertCameraImage(image);
      if (converted == null) {
        return;
      }

      final img.Image resized = img.copyResize(converted, width: 224, height: 224);
      final Float32List inputBytes = Float32List(1 * 224 * 224 * 3);

      int pixelIndex = 0;
      for (int y = 0; y < 224; y++) {
        for (int x = 0; x < 224; x++) {
          final img.Pixel pixel = resized.getPixel(x, y);
          inputBytes[pixelIndex++] = pixel.r / 255.0;
          inputBytes[pixelIndex++] = pixel.g / 255.0;
          inputBytes[pixelIndex++] = pixel.b / 255.0;
        }
      }

      final Object input = inputBytes.reshape(<int>[1, 224, 224, 3]);
      final List<List<double>> output = <List<double>>[
        List<double>.filled(_labels.length, 0.0),
      ];

      _interpreter!.run(input, output);

      int bestIndex = 0;
      double bestConfidence = output[0][0];
      for (int i = 1; i < output[0].length; i++) {
        if (output[0][i] > bestConfidence) {
          bestConfidence = output[0][i];
          bestIndex = i;
        }
      }

      final String label = _labels[bestIndex].trim();

      if (!_streamController.isClosed) {
        _lastEmittedAt = DateTime.now();
        _streamController.add(
          TFLiteResult(
            label: label,
            displayName: label,
            confidence: bestConfidence,
          ),
        );
      }
    } finally {
      _isProcessing = false;
    }
  }

  img.Image? _convertCameraImage(CameraImage image) {
    if (Platform.isAndroid && image.format.group == ImageFormatGroup.yuv420) {
      return _convertYUV420(image);
    }
    if (Platform.isIOS && image.format.group == ImageFormatGroup.bgra8888) {
      return img.Image.fromBytes(
        width: image.width,
        height: image.height,
        bytes: image.planes[0].bytes.buffer,
        order: img.ChannelOrder.bgra,
      );
    }
    return null;
  }

  img.Image _convertYUV420(CameraImage image) {
    final int width = image.width;
    final int height = image.height;
    final int uvRowStride = image.planes[1].bytesPerRow;
    final int uvPixelStride = image.planes[1].bytesPerPixel ?? 1;
    final img.Image rgbImage = img.Image(width: width, height: height);

    for (int y = 0; y < height; y++) {
      final int uvRow = y >> 1;
      for (int x = 0; x < width; x++) {
        final int uvCol = x >> 1;
        final int uvIndex = (uvRow * uvRowStride) + (uvCol * uvPixelStride);
        final int yIndex = y * image.planes[0].bytesPerRow + x;

        if (yIndex >= image.planes[0].bytes.length ||
            uvIndex >= image.planes[1].bytes.length ||
            uvIndex >= image.planes[2].bytes.length) {
          continue;
        }

        final int yp = image.planes[0].bytes[yIndex];
        final int up = image.planes[1].bytes[uvIndex];
        final int vp = image.planes[2].bytes[uvIndex];

        final int r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255);
        final int g =
            (yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91)
                .round()
                .clamp(0, 255);
        final int b = (yp + up * 1814 / 1024 - 227).round().clamp(0, 255);

        rgbImage.setPixelRgb(x, y, r, g, b);
      }
    }
    return rgbImage;
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _streamController.close();
  }
}
