import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:camera/camera.dart';
import 'package:gal/gal.dart';

import 'package:bigash/presentation/cubits/detection_cubit.dart';

class ScannerPage extends StatefulWidget {
  const ScannerPage({super.key});

  @override
  State<ScannerPage> createState() => _ScannerPageState();
}

class _ScannerPageState extends State<ScannerPage>
    with WidgetsBindingObserver {
  final GlobalKey _captureKey = GlobalKey();

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // Pause the camera + vibration + TTS when the app goes to background
    // so we don't burn battery, leak the camera handle, or crash on resume.
    final DetectionCubit cubit = context.read<DetectionCubit>();
    if (state == AppLifecycleState.paused ||
        state == AppLifecycleState.inactive ||
        state == AppLifecycleState.hidden) {
      cubit.pauseForBackground();
    } else if (state == AppLifecycleState.resumed) {
      cubit.resumeFromBackground();
    }
  }

  /// Avoid [CameraPreview] if the controller was disposed (stale state / race).
  static bool _canShowCameraPreview(CameraController? c, bool isActive) {
    if (c == null || !isActive) {
      return false;
    }
    try {
      return c.value.isInitialized;
    } catch (_) {
      return false;
    }
  }

  static String _semanticsLabel(DetectionState s) {
    if (!s.isActive) {
      return 'اضغط لتشغيل الكاميرا. ضغط مطوّل لالتقاط صورة الشاشة.';
    }
    switch (s.phase) {
      case CurrencyDetectionPhase.scanning:
        return 'مسح نشط مع اهتزاز. اضغط لإيقاف التعرف. ضغط مطوّل لالتقاط صورة الشاشة.';
      case CurrencyDetectionPhase.lowLight:
        return 'إضاءة منخفضة، الفلاش مفعّل. اضغط لإيقاف التعرف.';
      case CurrencyDetectionPhase.success:
        return 'تم التعرف على العملة. اضغط لإيقاف التعرف.';
      case CurrencyDetectionPhase.idle:
        return 'اضغط لإيقاف الكاميرا والتعرف';
    }
  }

  Future<void> _captureScreenshot() async {
    try {
      final RenderRepaintBoundary? boundary = _captureKey.currentContext
          ?.findRenderObject() as RenderRepaintBoundary?;
      if (boundary == null) return;

      final ui.Image image = await boundary.toImage(pixelRatio: 2.0);
      final ByteData? byteData =
          await image.toByteData(format: ui.ImageByteFormat.png);
      if (byteData == null) return;
      final Uint8List pngBytes = byteData.buffer.asUint8List();

      // Save directly into the phone's gallery under a "Bigash" album.
      final bool hasAccess = await Gal.hasAccess();
      if (!hasAccess) {
        final bool granted = await Gal.requestAccess();
        if (!granted) {
          if (!mounted) return;
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('تم رفض إذن الوصول إلى المعرض')),
          );
          return;
        }
      }

      final String filename =
          'bigash_${DateTime.now().millisecondsSinceEpoch}';
      await Gal.putImageBytes(pngBytes, album: 'Bigash', name: filename);

      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          duration: const Duration(seconds: 2),
          content: Text('حُفظت في معرض الصور: $filename.png'),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('فشل الالتقاط: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: BlocBuilder<DetectionCubit, DetectionState>(
        builder: (BuildContext context, DetectionState mlState) {
          final CameraController? controller = mlState.controller;
          final bool hasPreview = _canShowCameraPreview(controller, mlState.isActive);

          return Semantics(
            button: true,
            label: _semanticsLabel(mlState),
            child: GestureDetector(
              behavior: HitTestBehavior.opaque,
              onTap: () async {
                await context.read<DetectionCubit>().toggle();
              },
              onLongPress: _captureScreenshot,
              child: RepaintBoundary(
                key: _captureKey,
                child: Stack(
                  fit: StackFit.expand,
                  children: <Widget>[
                    if (hasPreview && controller != null)
                      CameraPreview(controller)
                    else
                      const ColoredBox(color: Colors.black),
                    Align(
                      alignment: Alignment.bottomCenter,
                      child: Container(
                        width: double.infinity,
                        padding: const EdgeInsets.all(20),
                        color: Colors.black.withValues(alpha: 0.45),
                        child: Text(
                          mlState.statusText,
                          textAlign: TextAlign.center,
                          style: const TextStyle(
                            color: Colors.yellow,
                            fontSize: 28,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}
