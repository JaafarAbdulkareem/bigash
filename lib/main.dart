import 'dart:async';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/semantics.dart';
import 'package:flutter/services.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:vibration/vibration.dart';
import 'package:image/image.dart' as img;

late List<CameraDescription> cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  try {
    cameras = await availableCameras();
  } on CameraException catch (e) {
    debugPrint('Error: ${e.code}\nError Message: ${e.description}');
  }
  runApp(const BigashApp());
}

class BigashApp extends StatelessWidget {
  const BigashApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Bigash',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        primaryColor: Colors.black,
        scaffoldBackgroundColor: Colors.black,
      ),
      home: const SplashScreen(),
    );
  }
}

class SplashScreen extends StatefulWidget {
  const SplashScreen({Key? key}) : super(key: key);

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  final FlutterTts flutterTts = FlutterTts();

  @override
  void initState() {
    super.initState();
    _initAndRoute();
  }

  Future<void> _initAndRoute() async {
    await flutterTts.setLanguage("ar-SA");
    await flutterTts.setSpeechRate(0.5);
    await flutterTts.setVolume(1.0);
    
    // Welcome message
    await flutterTts.speak("أهلاً بكم في تطبيق بِيْقَش");
    
    // Request permissions
    Map<Permission, PermissionStatus> statuses = await [
      Permission.camera,
      Permission.microphone,
    ].request();

    // Wait a brief moment to ensure TTS starts/finishes
    await Future.delayed(const Duration(seconds: 3));

    if (mounted) {
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (_) => const MainScreen()),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Semantics(
          label: "شاشة البداية، جاري تحميل تطبيق بيقش",
          child: const Text(
            'بيقش\nBigash',
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 48,
              fontWeight: FontWeight.bold,
              color: Colors.yellow,
            ),
          ),
        ),
      ),
    );
  }
}

class MainScreen extends StatefulWidget {
  const MainScreen({Key? key}) : super(key: key);

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  CameraController? _cameraController;
  Interpreter? _interpreter;
  List<String> _labels = [];
  bool _isProcessing = false;
  final FlutterTts flutterTts = FlutterTts();

  String _currentValue = "جاري التعرف...";
  String _lastSpokenValue = "";
  
  // To avoid constant TTS spamming, require same value multiple times or time delay
  DateTime _lastDetectionTime = DateTime.now();

  @override
  void initState() {
    super.initState();
    _initializeHardwareAndModel();
  }

  Future<void> _initializeHardwareAndModel() async {
    // 1. Long vibration on start
    bool? hasVib = await Vibration.hasVibrator();
    if (hasVib == true) {
      Vibration.vibrate(duration: 1000);
    }

    // 2. Setup TTS
    await flutterTts.setLanguage("ar-SA");
    await flutterTts.setSpeechRate(0.5);

    // 3. Load ML
    await _loadModelAndLabels();

    // 4. Setup Camera
    if (cameras.isNotEmpty) {
      _cameraController = CameraController(
        cameras[0],
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: Platform.isAndroid ? ImageFormatGroup.yuv420 : ImageFormatGroup.bgra8888,
      );

      await _cameraController!.initialize();
      
      // Auto flashlight logic
      _cameraController!.startImageStream((CameraImage image) {
        if (!_isProcessing) {
          _isProcessing = true;
          _processCameraImage(image);
        }
      });

      if (mounted) {
        setState(() {});
      }
    }
  }

  Future<void> _loadModelAndLabels() async {
    try {
      _interpreter = await Interpreter.fromAsset('lib/core/ai_model/model_unquant.tflite');
      String labelsData = await rootBundle.loadString('lib/core/ai_model/labels.txt');
      _labels = labelsData.split('\n').where((line) => line.trim().isNotEmpty).map((line) {
        // Line format: "0 50" -> we need "50"
        var parts = line.split(' ');
        if (parts.length > 1) {
          return parts.sublist(1).join(' ');
        }
        return line;
      }).toList();
    } catch (e) {
      debugPrint("Error loading model: $e");
    }
  }

  void _processCameraImage(CameraImage image) async {
    try {
      // Very basic luminance check for auto-flashlight
      // On Android YUV420, plane[0] is luminance (Y)
      if (Platform.isAndroid && image.format.group == ImageFormatGroup.yuv420) {
        int ySum = 0;
        final plane = image.planes[0].bytes;
        // Sample pixels for speed instead of summing all
        int sampleStep = 100;
        int count = 0;
        for (int i = 0; i < plane.length; i += sampleStep) {
          ySum += plane[i];
          count++;
        }
        double avgLuminance = ySum / count;
        if (avgLuminance < 40) { // arbitrary dark threshold
          _cameraController?.setFlashMode(FlashMode.torch);
        } else if (avgLuminance > 120) {
          _cameraController?.setFlashMode(FlashMode.off);
        }
      }

      if (_interpreter != null && _labels.isNotEmpty) {
        // Convert to image.Image and resize
        var inputImage = _convertCameraImage(image);
        if (inputImage == null) return;
        
        var resizedImage = img.copyResize(inputImage, width: 224, height: 224);
        
        // Normalize image data to [-1, 1] or [0, 1] depending on TM 
        // TM models usually use [0, 1] or [-1, 1] for unquantized. Let's assume TM standard [-1, 1]
        var inputBytes = Float32List(1 * 224 * 224 * 3);
        int pixelIndex = 0;
        for (int y = 0; y < 224; y++) {
          for (int x = 0; x < 224; x++) {
            var pixel = resizedImage.getPixel(x, y);
            inputBytes[pixelIndex++] = (pixel.r / 127.5) - 1.0;
            inputBytes[pixelIndex++] = (pixel.g / 127.5) - 1.0;
            inputBytes[pixelIndex++] = (pixel.b / 127.5) - 1.0;
          }
        }

        var input = inputBytes.reshape([1, 224, 224, 3]);
        var output = List<double>.filled(1 * _labels.length, 0).reshape([1, _labels.length]);
        
        _interpreter!.run(input, output);
        
        final List<double> result = List<double>.from(output[0]);
        double highestProb = 0.0;
        int highestIndex = -1;

        for (int i = 0; i < result.length; i++) {
          if (result[i] > highestProb) {
            highestProb = result[i];
            highestIndex = i;
          }
        }

        if (highestProb >= 0.85 && highestIndex != -1) {
          String labelInfo = _labels[highestIndex].toLowerCase();
          
          if (!labelInfo.contains("unknown") && labelInfo.trim() != "6") {
            String arabicLabel = _mapToArabic(labelInfo);
            
            setState(() {
              _currentValue = arabicLabel;
            });

            // Prevent spamming
            if (arabicLabel != _lastSpokenValue || DateTime.now().difference(_lastDetectionTime).inSeconds > 4) {
              _lastSpokenValue = arabicLabel;
              _lastDetectionTime = DateTime.now();
              
              _triggerSuccess();
              flutterTts.speak(arabicLabel);
            }
          }
        }
      }
    } catch (e) {
      debugPrint("Inference error: $e");
    } finally {
      _isProcessing = false;
    }
  }

  String _mapToArabic(String rawLabel) {
    if (rawLabel.contains("250")) return "مائتان وخمسون ريال";
    if (rawLabel.contains("500")) return "خمسمائة ريال";
    if (rawLabel.contains("1000") || rawLabel.contains("1,000")) return "ألف ريال";
    if (rawLabel.contains("50")) return "خمسون ريال";
    if (rawLabel.contains("100")) return "مائة ريال";
    if (rawLabel.contains("200")) return "مائتان ريال";
    return rawLabel;
  }

  void _triggerSuccess() async {
    bool? hasVib = await Vibration.hasVibrator();
    if (hasVib == true) {
      // Two short vibrations to indicate detection
      Vibration.vibrate(pattern: [0, 100, 100, 100]);
    }
  }

  void _handleScreenTap() {
    if (_lastSpokenValue.isNotEmpty) {
      flutterTts.speak(_lastSpokenValue);      
    } else {
      flutterTts.speak("لم يتم التعرف على أي عملة بعد.");
    }
  }

  img.Image? _convertCameraImage(CameraImage image) {
    if (Platform.isAndroid && image.format.group == ImageFormatGroup.yuv420) {
      return _convertYUV420(image);
    } else if (Platform.isIOS && image.format.group == ImageFormatGroup.bgra8888) {
      return _convertBGRA8888(image);
    }
    return null;
  }

  img.Image _convertBGRA8888(CameraImage image) {
    return img.Image.fromBytes(
      width: image.width,
      height: image.height,
      bytes: image.planes[0].bytes.buffer,
      order: img.ChannelOrder.bgra,
    );
  }

  img.Image _convertYUV420(CameraImage image) {
    final width = image.width;
    final height = image.height;
    final uvRowStride = image.planes[1].bytesPerRow;
    final uvPixelStride = image.planes[1].bytesPerPixel ?? 1;

    final result = img.Image(width: width, height: height);

    for (int y = 0; y < height; y++) {
      int uvRow = y >> 1;
      for (int x = 0; x < width; x++) {
        int uvCol = x >> 1;
        int uvIndex = (uvRow * uvRowStride) + (uvCol * uvPixelStride);
        
        final yIndex = y * image.planes[0].bytesPerRow + x;
        final index = yIndex;

        if (index >= image.planes[0].bytes.length || uvIndex >= image.planes[1].bytes.length) {
           break;
        }

        final yp = image.planes[0].bytes[index];
        final up = image.planes[1].bytes[uvIndex];
        final vp = image.planes[2].bytes[uvIndex];

        int r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255);
        int g = (yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91).round().clamp(0, 255);
        int b = (yp + up * 1814 / 1024 - 227).round().clamp(0, 255);

        result.setPixelRgb(x, y, r, g, b);
      }
    }
    return result;
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _interpreter?.close();
    flutterTts.stop();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Semantics(
        label: "اضغط بأي مكان على الشاشة لقراءة العملة الحالية. يتم فتح الكاميرا للتعرف تلقائيا.",
        button: true,
        child: GestureDetector(
          onTap: _handleScreenTap,
          behavior: HitTestBehavior.opaque,
          child: Container(
            color: Colors.black, // Pitch black for high contrast
            width: double.infinity,
            height: double.infinity,
            child: SafeArea(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                   const Icon(Icons.touch_app, color: Colors.yellow, size: 80),
                   const SizedBox(height: 30),
                   Text(
                    _currentValue,
                    textAlign: TextAlign.center,
                    style: const TextStyle(
                      fontSize: 48,
                      fontWeight: FontWeight.bold,
                      color: Colors.yellow, // High contrast yellow
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
