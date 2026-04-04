import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:camera/camera.dart';

import 'package:bigash/presentation/cubits/camera_toggle_cubit.dart';
import 'package:bigash/presentation/cubits/detection_cubit.dart';

class ScannerPage extends StatelessWidget {
  const ScannerPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: BlocBuilder<DetectionCubit, DetectionState>(
        builder: (BuildContext context, DetectionState mlState) {
          final CameraController? controller = mlState.controller;
          final bool hasPreview =
              controller != null && controller.value.isInitialized && mlState.isActive;

          return Semantics(
            button: true,
            label: mlState.isActive
                ? 'اضغط لإيقاف الكاميرا والتعرف'
                : 'اضغط لتشغيل الكاميرا والتعرف',
            child: GestureDetector(
              behavior: HitTestBehavior.opaque,
              onTap: () async {
                context.read<CameraToggleCubit>().toggle();
                await context.read<DetectionCubit>().toggle();
              },
              child: Stack(
                fit: StackFit.expand,
                children: <Widget>[
                  if (hasPreview)
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
