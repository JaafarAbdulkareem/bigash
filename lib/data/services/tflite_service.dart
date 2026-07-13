import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:bigash/domain/currency/frame_quality.dart';
import 'package:bigash/domain/currency/model_asset_contract.dart';

class TFLiteResult {
  final String label;
  final String displayName;
  final double confidence;
  final List<MapEntry<String, double>> topK;
  final CaptureQuality quality;
  final double brightness;
  final double sharpness;

  const TFLiteResult({
    required this.label,
    required this.displayName,
    required this.confidence,
    this.topK = const <MapEntry<String, double>>[],
    this.quality = CaptureQuality.usable,
    this.brightness = 0,
    this.sharpness = 0,
  });
}

/// Normalization mode must match how the model was trained.
/// - [raw]     : pass raw [0, 255] as float32 (bigash_v18 has Rescaling layer inside)
/// - [signed]  : (x - 127.5) / 127.5  -> [-1, 1]
/// - [unsigned]: x / 255                -> [ 0, 1]
/// - [imageNet]: (x/255 - mean) / std            (MobileNetV3, EfficientNet)
enum NormalizationMode { raw, signed, unsigned, imageNet }

class TFLiteService {
  static const String modelAssetPath = 'assets/bigash_v18.tflite';
  static const bool _diagnosticsEnabled = bool.fromEnvironment(
    'BIGASH_DIAGNOSTICS',
    defaultValue: !kReleaseMode,
  );

  /// Minimum spacing between *emitted* results. Field measurement on a
  /// 4 GB Xiaomi M2004J19C showed 77–117% CPU while scanning at the old 250 ms
  /// cadence, even on rejected frames, because full-frame YUV conversion runs
  /// before the quality gate. Two samples per second still confirms a result
  /// with three votes in about 1.5 seconds, while materially reducing heat,
  /// battery drain, and the chance that thermal throttling stalls the camera.
  /// With [useTTA] on, usable frames also run FOUR inferences, so the real
  /// cadence is max(500 ms, 4× inference time).
  static const Duration _minFrameInterval = Duration(milliseconds: 500);

  /// The bigash model has its own Rescaling layer baked in, so it expects
  /// raw [0, 255] float32 pixel values — no Flutter-side normalization.
  NormalizationMode normalization = NormalizationMode.raw;

  /// Test-time augmentation: average inference over 4 rotations. ON — the
  /// model is trained with label smoothing + MixUp, so its softmax is soft;
  /// averaging rotations lifts the confidence enough that good frames
  /// actually clear the voting threshold. Worth the 4x inference cost.
  bool useTTA = true;

  Interpreter? _interpreter;
  List<String> _labels = <String>[];
  bool _isProcessing = false;
  DateTime? _lastEmittedAt;

  int _inputWidth = 0;
  int _inputHeight = 0;
  int _inputChannels = 0;
  int _inputBatch = 0;
  String _inputTypeName = '';
  String _outputTypeName = '';
  bool _inputIsUint8 = false;
  double _inputScale = 1.0;
  int _inputZeroPoint = 0;
  bool _outputIsUint8 = false;
  double _outputScale = 1.0;
  int _outputZeroPoint = 0;
  int _numClasses = 0;

  /// Non-null when the model/labels CONTRACT is violated (model loaded but its
  /// output class count disagrees with assets/labels.txt, or labels failed to
  /// load). A spoken-ready Arabic message — surfaced to the blind user instead
  /// of silently mis-announcing every bill.
  String? _contractError;
  String? _assetIntegrityError;

  final StreamController<TFLiteResult> _streamController =
      StreamController<TFLiteResult>.broadcast();

  Stream<TFLiteResult> get resultStream => _streamController.stream;

  bool get isModelLoaded => _interpreter != null;

  String? get contractError => _contractError;

  /// True only when the model is loaded, labels loaded, and the model's class
  /// count matches the label count. The decision path must never run unless
  /// this holds — a label/class mismatch would silently mis-map denominations.
  bool get isModelReady =>
      _interpreter != null && _labels.isNotEmpty && _contractError == null;

  Future<void> initialize() async {
    await _verifyBundledAssets();

    if (_assetIntegrityError == null && _interpreter == null) {
      try {
        _interpreter = await Interpreter.fromAsset(modelAssetPath);
        _readTensorShapes();
      } on FlutterError catch (e) {
        debugPrint('TFLite model not bundled or load failed: $e');
      } catch (e) {
        debugPrint('TFLite model load failed: $e');
      }
    }

    _validateContract();
  }

  Future<void> _verifyBundledAssets() async {
    _assetIntegrityError = null;
    try {
      final String labelsContent = await rootBundle.loadString(
        'assets/labels.txt',
      );
      _labels = labelsContent
          .split('\n')
          .map((String e) => e.trim())
          .where((String e) => e.isNotEmpty)
          .toList(growable: false);
      if (!labelsMatchExpectedBigashOrder(_labels)) {
        _assetIntegrityError =
            'ترتيب تسميات النموذج غير معتمد. أعد بناء التطبيق بالنموذج والتسميات المراجعة.';
        return;
      }

      final ByteData modelData = await rootBundle.load(modelAssetPath);
      final Uint8List modelBytes = modelData.buffer.asUint8List(
        modelData.offsetInBytes,
        modelData.lengthInBytes,
      );
      final String actualDigest = sha256.convert(modelBytes).toString();
      if (actualDigest != expectedBigashModelSha256) {
        _assetIntegrityError =
            'ملف نموذج التعرف غير معتمد أو تالف. أعد تثبيت نسخة رسمية من التطبيق.';
        debugPrint(
          '[TFLite] MODEL DIGEST MISMATCH: expected='
          '$expectedBigashModelSha256 actual=$actualDigest',
        );
      }
    } catch (e) {
      _labels = <String>[];
      _assetIntegrityError =
          'تعذّر التحقق من ملفات نموذج التعرف. أعد تثبيت التطبيق.';
      debugPrint('[TFLite] asset integrity check failed: $e');
    }
  }

  /// Verifies the bundled model and labels agree. A mismatch (e.g. a retrain
  /// changed the class count but labels.txt wasn't updated, or vice-versa)
  /// would otherwise be hidden by the `min(probs, labels)` clamp in
  /// [processImage] and silently mis-announce denominations — the catastrophic
  /// failure for this app. We fail LOUD: [isModelReady] goes false and
  /// [contractError] carries a spoken Arabic explanation.
  void _validateContract() {
    _contractError = _assetIntegrityError;
    if (_contractError != null) return;
    if (_interpreter == null) {
      return; // "no model" is handled separately (generic message).
    }
    if (_labels.isEmpty) {
      _contractError =
          'تعذّر تحميل ملف التسميات. أعد بناء التطبيق مع ملف labels المطابق للنموذج.';
      return;
    }
    final bool supportedInputType =
        _inputTypeName.contains('float32') || _inputTypeName.contains('uint8');
    final bool supportedOutputType =
        _outputTypeName.contains('float32') ||
        _outputTypeName.contains('uint8');
    if (_numClasses <= 0 ||
        _inputWidth <= 0 ||
        _inputHeight <= 0 ||
        _inputBatch != 1 ||
        _inputChannels != 3 ||
        !supportedInputType ||
        !supportedOutputType) {
      _contractError =
          'بنية النموذج غير مدعومة. أعد تصدير نموذج بصور RGB ومخارج متوافقة ثم أعد بناء التطبيق.';
      debugPrint(
        '[TFLite] INVALID TENSORS: input=$_inputBatch x '
        '${_inputWidth}x$_inputHeight x $_inputChannels ($_inputTypeName), '
        'classes=$_numClasses ($_outputTypeName)',
      );
      return;
    }
    if (_numClasses > 0 && _numClasses != _labels.length) {
      _contractError =
          'النموذج وملف التسميات غير متطابقين. أعد بناء التطبيق بنموذج وتسميات من نفس التدريب.';
      debugPrint(
        '[TFLite] CONTRACT MISMATCH: model classes=$_numClasses, labels=${_labels.length}',
      );
    }
  }

  void _readTensorShapes() {
    final interpreter = _interpreter;
    if (interpreter == null) return;

    try {
      final inputTensor = interpreter.getInputTensor(0);
      final shape = inputTensor.shape; // [1, H, W, 3]
      if (shape.length == 4) {
        _inputBatch = shape[0];
        _inputHeight = shape[1];
        _inputWidth = shape[2];
        _inputChannels = shape[3];
      }
      _inputTypeName = inputTensor.type.toString().toLowerCase();
      _inputIsUint8 = _inputTypeName.contains('uint8');
      final inQp = inputTensor.params;
      _inputScale = inQp.scale == 0 ? 1.0 : inQp.scale;
      _inputZeroPoint = inQp.zeroPoint;

      final outputTensor = interpreter.getOutputTensor(0);
      final outShape = outputTensor.shape;
      _numClasses = outShape.last;
      _outputTypeName = outputTensor.type.toString().toLowerCase();
      _outputIsUint8 = _outputTypeName.contains('uint8');

      final qp = outputTensor.params;
      _outputScale = qp.scale == 0 ? 1.0 : qp.scale;
      _outputZeroPoint = qp.zeroPoint;

      if (_diagnosticsEnabled) {
        debugPrint(
          '[TFLite] input=$shape uint8=$_inputIsUint8 '
          'inScale=$_inputScale inZp=$_inputZeroPoint, output=$outShape '
          'uint8=$_outputIsUint8 outScale=$_outputScale '
          'outZp=$_outputZeroPoint',
        );
      }
    } catch (e) {
      debugPrint('[TFLite] could not read tensor shape: $e');
    }
  }

  /// [sensorOrientation] comes from [CameraDescription.sensorOrientation]
  /// (0, 90, 180, 270). The raw frame must be rotated by this amount so
  /// the model sees an upright image, matching its training data.
  Future<void> processImage(
    CameraImage image, {
    int sensorOrientation = 90,
    bool mirror = false,
  }) async {
    // Never run the decision path on a missing model, missing labels, or a
    // model/labels contract mismatch — any of those could mis-map denominations.
    if (_isProcessing || !isModelReady) {
      return;
    }

    final DateTime now = DateTime.now();
    if (_lastEmittedAt != null &&
        now.difference(_lastEmittedAt!) < _minFrameInterval) {
      return;
    }

    _isProcessing = true;
    try {
      img.Image? frame = _convertCameraImage(image);
      if (frame == null) {
        return;
      }

      // 1) Rotate to upright (correct sensor orientation).
      if (sensorOrientation != 0) {
        frame = img.copyRotate(frame, angle: sensorOrientation);
      }
      if (mirror) {
        frame = img.flipHorizontal(frame);
      }

      // 2) Center-crop to square (preserve aspect ratio).
      final int side = math.min(frame.width, frame.height);
      final int offX = (frame.width - side) ~/ 2;
      final int offY = (frame.height - side) ~/ 2;
      final img.Image cropped = img.copyCrop(
        frame,
        x: offX,
        y: offY,
        width: side,
        height: side,
      );

      // 3) Resize to model input.
      final img.Image resized = img.copyResize(
        cropped,
        width: _inputWidth,
        height: _inputHeight,
      );

      // Reject frames whose detail or exposure is too poor before they can
      // contribute a dangerous denomination vote. This is especially important
      // when a blind user holds the phone by hand and motion blur is likely.
      final FrameQualityMetrics qualityMetrics = _measureFrameQuality(resized);
      final CaptureQuality quality = assessFrameQuality(qualityMetrics);
      if (quality != CaptureQuality.usable) {
        _emitQualityResult(quality, qualityMetrics);
        return;
      }

      // 4) Build TTA variants. Rotations (not flips). The v18 training applied
      // orthogonal 90°/180°/270° rotation to only ~30% of TRAIN images and kept
      // VAL upright, so the upright view is the best-calibrated input and the
      // rotations are a robustness aid — averaging them lifts the soft model's
      // confidence enough that good frames clear the voting threshold. Flip was
      // NOT in training augmentation, so we don't use it. (If field data shows
      // TTA lowers the clear-rate for upright bills, drop 180/270 or weight the
      // upright variant higher.)
      final List<img.Image> variants = useTTA
          ? <img.Image>[
              resized,
              img.copyRotate(resized, angle: 90),
              img.copyRotate(resized, angle: 180),
              img.copyRotate(resized, angle: 270),
            ]
          : <img.Image>[resized];

      // 5) Run inference on each variant and accumulate probabilities.
      // Skip any variant whose output contains NaN/Inf — that would
      // corrupt the average and silently mis-classify.
      final List<double> accumProbs = List<double>.filled(_numClasses, 0.0);
      int validVariants = 0;
      for (final img.Image variant in variants) {
        final List<double> probs = _runInferenceOnce(variant);
        if (_hasBadValues(probs)) {
          debugPrint('[TFLite] skipping variant with NaN/Inf output');
          continue;
        }
        for (int i = 0; i < _numClasses; i++) {
          accumProbs[i] += probs[i];
        }
        validVariants++;
      }
      if (validVariants == 0) {
        debugPrint(
          '[TFLite] all TTA variants produced bad outputs — dropping frame',
        );
        return;
      }
      for (int i = 0; i < _numClasses; i++) {
        accumProbs[i] /= validVariants;
      }

      // 6) Pick the best class and top-3.
      final int n = math.min(accumProbs.length, _labels.length);
      final List<int> indices = List<int>.generate(n, (i) => i)
        ..sort((a, b) => accumProbs[b].compareTo(accumProbs[a]));

      final int bestIndex = indices.first;
      final double bestConfidence = accumProbs[bestIndex];
      // Defensive: if even the averaged best is broken, drop the frame.
      if (bestConfidence.isNaN || bestConfidence.isInfinite) {
        debugPrint('[TFLite] best confidence is NaN/Inf — dropping frame');
        return;
      }
      final String label = _labels[bestIndex].trim();

      // Carry the FULL class distribution (all classes), not just the top-3.
      // The decision core's out-of-distribution gate sums the `other` mass and
      // the new/old sub-variant mass from THIS list (frame_decision.dart). If we
      // truncated to the top-3, a foreign note whose `other` probability ranks
      // 4th+ — easy, because y100/y200 each occupy two classes that can fill the
      // top slots — would reach classifyFrame with otherProb=0 and be announced
      // as a confident Yemeni denomination, the exact safety failure the gate
      // exists to prevent. (It also under-counted a worn bill whose two designs
      // were split by a distractor.) Truncation is for display only — see
      // DetectionCubit._buildDebugText.
      final List<MapEntry<String, double>> topK = <MapEntry<String, double>>[];
      for (int i = 0; i < indices.length; i++) {
        topK.add(MapEntry(_labels[indices[i]].trim(), accumProbs[indices[i]]));
      }
      if (_diagnosticsEnabled) {
        debugPrint(
          '[TFLite] ${normalization.name} '
          'TTA=${useTTA ? "x${variants.length}" : "off"} top='
          '${topK.take(3).map((e) => "${e.key}=${e.value.toStringAsFixed(2)}").join(", ")}',
        );
      }

      if (!_streamController.isClosed) {
        _lastEmittedAt = DateTime.now();
        _streamController.add(
          TFLiteResult(
            label: label,
            displayName: label,
            confidence: bestConfidence,
            topK: topK,
          ),
        );
      }
    } catch (e, st) {
      debugPrint('[TFLite] inference failed: $e\n$st');
    } finally {
      _isProcessing = false;
    }
  }

  /// Runs one forward pass on the given image and returns softmaxed probs.
  List<double> _runInferenceOnce(img.Image src) {
    final Object input = _buildInputTensor(src);

    List<double> probs;
    if (_outputIsUint8) {
      final output = List.generate(1, (_) => List<int>.filled(_numClasses, 0));
      _interpreter!.run(input, output);
      probs = output[0]
          .map<double>((int q) => (q - _outputZeroPoint) * _outputScale)
          .toList();
    } else {
      final output = List.generate(
        1,
        (_) => List<double>.filled(_numClasses, 0.0),
      );
      _interpreter!.run(input, output);
      probs = List<double>.from(output[0]);
    }

    return _maybeSoftmax(probs);
  }

  Object _buildInputTensor(img.Image src) {
    final int w = src.width;
    final int h = src.height;

    if (_inputIsUint8) {
      final Uint8List bytes = Uint8List(1 * h * w * 3);
      int i = 0;
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          final img.Pixel p = src.getPixel(x, y);
          bytes[i++] = _quantizeInput(
            _normalizeChannel(p.r.toDouble(), channel: 0),
          );
          bytes[i++] = _quantizeInput(
            _normalizeChannel(p.g.toDouble(), channel: 1),
          );
          bytes[i++] = _quantizeInput(
            _normalizeChannel(p.b.toDouble(), channel: 2),
          );
        }
      }
      return bytes.reshape(<int>[1, h, w, 3]);
    }

    final Float32List bytes = Float32List(1 * h * w * 3);
    int i = 0;
    switch (normalization) {
      case NormalizationMode.raw:
        for (int y = 0; y < h; y++) {
          for (int x = 0; x < w; x++) {
            final img.Pixel p = src.getPixel(x, y);
            bytes[i++] = p.r.toDouble();
            bytes[i++] = p.g.toDouble();
            bytes[i++] = p.b.toDouble();
          }
        }
        break;
      case NormalizationMode.signed:
        for (int y = 0; y < h; y++) {
          for (int x = 0; x < w; x++) {
            final img.Pixel p = src.getPixel(x, y);
            bytes[i++] = (p.r - 127.5) / 127.5;
            bytes[i++] = (p.g - 127.5) / 127.5;
            bytes[i++] = (p.b - 127.5) / 127.5;
          }
        }
        break;
      case NormalizationMode.unsigned:
        for (int y = 0; y < h; y++) {
          for (int x = 0; x < w; x++) {
            final img.Pixel p = src.getPixel(x, y);
            bytes[i++] = p.r / 255.0;
            bytes[i++] = p.g / 255.0;
            bytes[i++] = p.b / 255.0;
          }
        }
        break;
      case NormalizationMode.imageNet:
        const double rMean = 0.485, gMean = 0.456, bMean = 0.406;
        const double rStd = 0.229, gStd = 0.224, bStd = 0.225;
        for (int y = 0; y < h; y++) {
          for (int x = 0; x < w; x++) {
            final img.Pixel p = src.getPixel(x, y);
            bytes[i++] = (p.r / 255.0 - rMean) / rStd;
            bytes[i++] = (p.g / 255.0 - gMean) / gStd;
            bytes[i++] = (p.b / 255.0 - bMean) / bStd;
          }
        }
        break;
    }
    return bytes.reshape(<int>[1, h, w, 3]);
  }

  void _emitQualityResult(CaptureQuality quality, FrameQualityMetrics metrics) {
    if (_streamController.isClosed) return;
    _lastEmittedAt = DateTime.now();
    _streamController.add(
      TFLiteResult(
        label: '',
        displayName: '',
        confidence: 0,
        quality: quality,
        brightness: metrics.meanLuma,
        sharpness: metrics.meanGradient,
      ),
    );
  }

  FrameQualityMetrics _measureFrameQuality(img.Image image) {
    const int step = 4;
    double lumaTotal = 0;
    double gradientTotal = 0;
    int samples = 0;
    int gradientSamples = 0;
    int darkSamples = 0;
    int brightSamples = 0;

    double lumaAt(int x, int y) {
      final img.Pixel p = image.getPixel(x, y);
      return 0.299 * p.r + 0.587 * p.g + 0.114 * p.b;
    }

    for (int y = 0; y < image.height; y += step) {
      for (int x = 0; x < image.width; x += step) {
        final double luma = lumaAt(x, y);
        lumaTotal += luma;
        samples++;
        if (luma < 25) darkSamples++;
        if (luma > 245) brightSamples++;

        if (x >= step) {
          gradientTotal += (luma - lumaAt(x - step, y)).abs();
          gradientSamples++;
        }
        if (y >= step) {
          gradientTotal += (luma - lumaAt(x, y - step)).abs();
          gradientSamples++;
        }
      }
    }

    return FrameQualityMetrics(
      meanLuma: samples == 0 ? 0 : lumaTotal / samples,
      meanGradient: gradientSamples == 0 ? 0 : gradientTotal / gradientSamples,
      darkRatio: samples == 0 ? 1 : darkSamples / samples,
      brightRatio: samples == 0 ? 1 : brightSamples / samples,
    );
  }

  double _normalizeChannel(double value, {required int channel}) {
    switch (normalization) {
      case NormalizationMode.raw:
        return value;
      case NormalizationMode.signed:
        return (value - 127.5) / 127.5;
      case NormalizationMode.unsigned:
        return value / 255.0;
      case NormalizationMode.imageNet:
        const List<double> means = <double>[0.485, 0.456, 0.406];
        const List<double> stds = <double>[0.229, 0.224, 0.225];
        return (value / 255.0 - means[channel]) / stds[channel];
    }
  }

  int _quantizeInput(double value) {
    return (value / _inputScale + _inputZeroPoint).round().clamp(0, 255);
  }

  /// Returns true if any entry is NaN or infinite. NaN slipping into the
  /// average would corrupt the entire top-K and silently mis-classify.
  bool _hasBadValues(List<double> values) {
    for (final double v in values) {
      if (v.isNaN || v.isInfinite) return true;
    }
    return false;
  }

  List<double> _maybeSoftmax(List<double> values) {
    double sum = 0;
    bool looksProb = true;
    for (final v in values) {
      sum += v;
      if (v < -0.01 || v > 1.01) {
        looksProb = false;
      }
    }
    if (looksProb && (sum > 0.9 && sum < 1.1)) {
      return values;
    }
    double maxV = values[0];
    for (final v in values) {
      if (v > maxV) maxV = v;
    }
    double denom = 0;
    final exps = List<double>.filled(values.length, 0);
    for (int i = 0; i < values.length; i++) {
      final e = math.exp(values[i] - maxV);
      exps[i] = e;
      denom += e;
    }
    if (denom == 0) return values;
    for (int i = 0; i < exps.length; i++) {
      exps[i] /= denom;
    }
    return exps;
  }

  img.Image? _convertCameraImage(CameraImage image) {
    if (Platform.isAndroid && image.format.group == ImageFormatGroup.yuv420) {
      return _convertYUV420(image);
    }
    if (Platform.isIOS && image.format.group == ImageFormatGroup.bgra8888) {
      // iOS CVPixelBuffer rows are frequently padded so bytesPerRow > width*4
      // (alignment), and the plane view may start at a non-zero offset. Passing
      // rowStride + bytesOffset is REQUIRED — without them image.fromBytes
      // assumes tightly-packed rows and reads every row after the first from
      // the wrong offset, shearing the frame diagonally before crop/resize and
      // silently corrupting the prediction.
      final Plane plane = image.planes[0];
      return img.Image.fromBytes(
        width: image.width,
        height: image.height,
        bytes: plane.bytes.buffer,
        bytesOffset: plane.bytes.offsetInBytes,
        rowStride: plane.bytesPerRow,
        order: img.ChannelOrder.bgra,
      );
    }
    debugPrint(
      '[TFLite] unsupported camera frame format: ${image.format.group} '
      '— frame skipped',
    );
    return null;
  }

  img.Image _convertYUV420(CameraImage image) {
    final int width = image.width;
    final int height = image.height;
    final int yRowStride = image.planes[0].bytesPerRow;
    final int uvRowStride = image.planes[1].bytesPerRow;
    final int uvPixelStride = image.planes[1].bytesPerPixel ?? 1;

    final Uint8List yPlane = image.planes[0].bytes;
    final Uint8List uPlane = image.planes[1].bytes;
    final Uint8List vPlane = image.planes[2].bytes;

    final img.Image rgbImage = img.Image(width: width, height: height);

    for (int y = 0; y < height; y++) {
      final int uvRow = y >> 1;
      for (int x = 0; x < width; x++) {
        final int uvCol = x >> 1;
        final int uvIndex = (uvRow * uvRowStride) + (uvCol * uvPixelStride);
        final int yIndex = y * yRowStride + x;

        if (yIndex >= yPlane.length ||
            uvIndex >= uPlane.length ||
            uvIndex >= vPlane.length) {
          continue;
        }

        final int yp = yPlane[yIndex];
        final int up = uPlane[uvIndex] - 128;
        final int vp = vPlane[uvIndex] - 128;

        // BT.601 YUV -> RGB.
        int r = (yp + 1.402 * vp).round();
        int g = (yp - 0.344136 * up - 0.714136 * vp).round();
        int b = (yp + 1.772 * up).round();

        r = r.clamp(0, 255);
        g = g.clamp(0, 255);
        b = b.clamp(0, 255);

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
