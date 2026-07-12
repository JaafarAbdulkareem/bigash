import 'package:bigash/presentation/cubits/detection_cubit.dart';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

/// Full-screen scanner UI without plugin calls or state mutation.
class ScannerSurface extends StatelessWidget {
  const ScannerSurface({required this.state, required this.onTap, super.key});

  final DetectionState state;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final CameraController? controller = state.controller;
    final bool hasPreview = _canShowCameraPreview(controller, state.isActive);

    return Semantics(
      container: true,
      excludeSemantics: true,
      button: true,
      label: scannerSemanticsLabel(state),
      child: GestureDetector(
        key: const Key('scanner_action_surface'),
        behavior: HitTestBehavior.opaque,
        onTap: onTap,
        child: Stack(
          fit: StackFit.expand,
          children: <Widget>[
            if (hasPreview && controller != null)
              CameraPreview(controller)
            else
              const ColoredBox(color: Colors.black),
            ScannerPhaseBorder(state: state),
            if (state.isActive && state.debugText.isNotEmpty)
              ScannerDebugOverlay(text: state.debugText),
            ScannerStatusOverlay(state: state),
          ],
        ),
      ),
    );
  }
}

class ScannerPhaseBorder extends StatelessWidget {
  const ScannerPhaseBorder({required this.state, super.key});

  final DetectionState state;

  @override
  Widget build(BuildContext context) {
    return IgnorePointer(
      child: AnimatedContainer(
        key: const Key('scanner_phase_border'),
        duration: const Duration(milliseconds: 250),
        curve: Curves.easeOut,
        decoration: BoxDecoration(
          border: Border.all(color: scannerPhaseColor(state), width: 6),
        ),
      ),
    );
  }
}

class ScannerDebugOverlay extends StatelessWidget {
  const ScannerDebugOverlay({required this.text, super.key});

  final String text;

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: Alignment.topCenter,
      child: SafeArea(
        bottom: false,
        child: Container(
          key: const Key('scanner_debug_overlay'),
          width: double.infinity,
          padding: const EdgeInsets.all(12),
          color: Colors.black.withValues(alpha: 0.55),
          child: Text(
            text,
            textAlign: TextAlign.center,
            style: const TextStyle(
              color: Colors.cyanAccent,
              fontSize: 16,
              fontFamily: 'monospace',
            ),
          ),
        ),
      ),
    );
  }
}

class ScannerStatusOverlay extends StatelessWidget {
  const ScannerStatusOverlay({required this.state, super.key});

  final DetectionState state;

  @override
  Widget build(BuildContext context) {
    final Color phaseColor = scannerPhaseColor(state);
    return Align(
      alignment: Alignment.bottomCenter,
      child: SafeArea(
        top: false,
        child: AnimatedContainer(
          key: const Key('scanner_status_overlay'),
          duration: const Duration(milliseconds: 250),
          curve: Curves.easeOut,
          width: double.infinity,
          padding: const EdgeInsets.fromLTRB(20, 24, 20, 20),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topCenter,
              end: Alignment.bottomCenter,
              colors: <Color>[
                Colors.black.withValues(alpha: 0),
                Colors.black.withValues(alpha: 0.75),
              ],
            ),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              Icon(
                scannerPhaseIcon(state),
                color: phaseColor == Colors.transparent
                    ? Colors.yellow
                    : phaseColor,
                size: 32,
              ),
              const SizedBox(width: 12),
              Flexible(
                child: Text(
                  state.statusText,
                  key: const Key('scanner_status_text'),
                  textAlign: TextAlign.center,
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
    );
  }
}

bool _canShowCameraPreview(CameraController? controller, bool isActive) {
  if (controller == null || !isActive) return false;
  try {
    return controller.value.isInitialized;
  } catch (_) {
    return false;
  }
}

Color scannerPhaseColor(DetectionState state) {
  if (!state.isActive) return Colors.transparent;
  switch (state.phase) {
    case CurrencyDetectionPhase.scanning:
      return const Color(0xFFFFB300);
    case CurrencyDetectionPhase.lowLight:
      return const Color(0xFFE53935);
    case CurrencyDetectionPhase.success:
      return const Color(0xFF43A047);
    case CurrencyDetectionPhase.idle:
      return Colors.transparent;
  }
}

IconData scannerPhaseIcon(DetectionState state) {
  if (!state.isActive) return Icons.touch_app_outlined;
  switch (state.phase) {
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

String scannerSemanticsLabel(DetectionState state) {
  if (!state.isActive) {
    return 'اضغط لتشغيل الكاميرا.';
  }
  switch (state.phase) {
    case CurrencyDetectionPhase.scanning:
      return 'مسح نشط مع اهتزاز. اضغط لإيقاف التعرف.';
    case CurrencyDetectionPhase.lowLight:
      return 'الصورة غير واضحة، الفلاش مفعّل. اضغط لإيقاف التعرف.';
    case CurrencyDetectionPhase.success:
      return '${state.statusText}. اضغط لإيقاف التعرف.';
    case CurrencyDetectionPhase.idle:
      return 'اضغط لإيقاف الكاميرا والتعرف.';
  }
}
