import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:camera/camera.dart';
import 'package:gal/gal.dart';
import 'package:vibration/vibration.dart';

import 'package:bigash/core/di/injection_container.dart' as di;
import 'package:bigash/data/services/tts_service.dart';
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
    // The scanner screen is silent until tapped — once the first frame is
    // drawn, tell the blind user out loud what to do.
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) {
        context.read<DetectionCubit>().announceIntro();
      }
    });
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

  /// Color of the full-screen phase border. Invisible to blind users (they
  /// rely on TTS + haptics) but gives partially-sighted users and helpers
  /// an instant, peripheral-vision read of the current detection state.
  /// Kept dark when not actively scanning so the camera preview is unframed.
  static Color _phaseColor(DetectionState s) {
    if (!s.isActive) {
      return Colors.transparent;
    }
    switch (s.phase) {
      case CurrencyDetectionPhase.scanning:
        return const Color(0xFFFFB300); // amber 600
      case CurrencyDetectionPhase.lowLight:
        return const Color(0xFFE53935); // red 600
      case CurrencyDetectionPhase.success:
        return const Color(0xFF43A047); // green 600
      case CurrencyDetectionPhase.idle:
        return Colors.transparent;
    }
  }

  /// Icon shown next to the status bar text. Same purpose as `_phaseColor`:
  /// instant visual phase cue for partially-sighted users and helpers.
  static IconData _phaseIcon(DetectionState s) {
    if (!s.isActive) {
      return Icons.touch_app_outlined;
    }
    switch (s.phase) {
      case CurrencyDetectionPhase.scanning:
        return Icons.center_focus_strong_outlined;
      case CurrencyDetectionPhase.lowLight:
        return Icons.flashlight_on_outlined;
      case CurrencyDetectionPhase.success:
        return Icons.check_circle_outline;
      case CurrencyDetectionPhase.idle:
        return Icons.touch_app_outlined;
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

  /// Audio + haptic feedback for the screenshot flow. A blind user can't see a
  /// SnackBar, so every outcome (saved / denied / failed) is also spoken and
  /// vibrated. Uses the same singleton TTS as the rest of the app.
  void _screenshotFeedback(String spoken, {bool error = false}) {
    unawaited(di.sl<TTSService>().speak(spoken));
    unawaited(
      Vibration.hasVibrator().then((bool has) {
        if (has) {
          Vibration.vibrate(
            pattern: error
                ? <int>[0, 300, 150, 300]
                : <int>[0, 120],
          );
        }
      }),
    );
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
          _screenshotFeedback('تم رفض إذن الوصول إلى المعرض', error: true);
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

      _screenshotFeedback('تم حفظ الصورة في المعرض');
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          duration: const Duration(seconds: 2),
          content: Text('حُفظت في معرض الصور: $filename.png'),
        ),
      );
    } catch (e) {
      _screenshotFeedback('فشل التقاط الصورة', error: true);
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
                    // Phase border — 6px frame around the whole screen that
                    // animates between amber / red / green so partially-
                    // sighted users and helpers can read the detection state
                    // at a glance. IgnorePointer keeps the gesture surface
                    // fully reachable underneath.
                    IgnorePointer(
                      child: AnimatedContainer(
                        duration: const Duration(milliseconds: 250),
                        curve: Curves.easeOut,
                        decoration: BoxDecoration(
                          border: Border.all(
                            color: _phaseColor(mlState),
                            width: 6,
                          ),
                        ),
                      ),
                    ),
                    // Live top-3 readout — invisible to a blind user; kept
                    // for field calibration of the voting thresholds.
                    if (mlState.isActive && mlState.debugText.isNotEmpty)
                      Align(
                        alignment: Alignment.topCenter,
                        child: Container(
                          width: double.infinity,
                          padding: const EdgeInsets.fromLTRB(12, 48, 12, 12),
                          color: Colors.black.withValues(alpha: 0.55),
                          child: Text(
                            mlState.debugText,
                            textAlign: TextAlign.center,
                            style: const TextStyle(
                              color: Colors.cyanAccent,
                              fontSize: 16,
                              fontFamily: 'monospace',
                            ),
                          ),
                        ),
                      ),
                    Align(
                      alignment: Alignment.bottomCenter,
                      child: AnimatedContainer(
                        duration: const Duration(milliseconds: 250),
                        curve: Curves.easeOut,
                        width: double.infinity,
                        padding: const EdgeInsets.fromLTRB(20, 24, 20, 28),
                        decoration: BoxDecoration(
                          // Vertical gradient instead of a flat translucent
                          // bar — readable over both bright and dark camera
                          // frames without obscuring the bill.
                          gradient: LinearGradient(
                            begin: Alignment.topCenter,
                            end: Alignment.bottomCenter,
                            colors: <Color>[
                              Colors.black.withValues(alpha: 0.0),
                              Colors.black.withValues(alpha: 0.75),
                            ],
                          ),
                        ),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: <Widget>[
                            // A small phase glyph beside the status — the
                            // blind user never sees it, but it helps a
                            // helper or partially-sighted user identify the
                            // phase faster than reading words.
                            Icon(
                              _phaseIcon(mlState),
                              color: _phaseColor(mlState) == Colors.transparent
                                  ? Colors.yellow
                                  : _phaseColor(mlState),
                              size: 32,
                            ),
                            const SizedBox(width: 12),
                            Flexible(
                              child: Text(
                                mlState.statusText,
                                textAlign: TextAlign.center,
                                // Cap lines + ellipsis so a long hint at a
                                // large accessibility text scale can't overflow
                                // and break the layout. The status is always
                                // ALSO spoken, so visual truncation in extreme
                                // scales is safe.
                                maxLines: 4,
                                overflow: TextOverflow.ellipsis,
                                style: const TextStyle(
                                  color: Colors.yellow,
                                  fontSize: 28,
                                  fontWeight: FontWeight.bold,
                                  height: 1.2,
                                ),
                              ),
                            ),
                          ],
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
