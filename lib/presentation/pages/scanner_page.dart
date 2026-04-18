import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:camera/camera.dart';

import 'package:bigash/presentation/cubits/camera_toggle_cubit.dart';
import 'package:bigash/presentation/cubits/detection_cubit.dart';

class ScannerPage extends StatelessWidget {
  const ScannerPage({super.key});

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
      return 'اضغط لتشغيل الكاميرا والتعرف';
    }
    switch (s.phase) {
      case CurrencyDetectionPhase.scanning:
        return 'مسح نشط مع اهتزاز. اضغط لإيقاف التعرف.';
      case CurrencyDetectionPhase.lowLight:
        return 'إضاءة منخفضة، الفلاش مفعّل. اضغط لإيقاف التعرف.';
      case CurrencyDetectionPhase.success:
        return 'تم التعرف على العملة. اضغط لإيقاف التعرف.';
      case CurrencyDetectionPhase.idle:
        return 'اضغط لإيقاف الكاميرا والتعرف';
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
                context.read<CameraToggleCubit>().toggle();
                await context.read<DetectionCubit>().toggle();
              },
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
          );
        },
      ),
    );
  }
}
