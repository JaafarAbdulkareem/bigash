import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import 'package:bigash/presentation/cubits/detection_cubit.dart';
import 'package:bigash/presentation/widgets/scanner_surface.dart';

class ScannerPage extends StatefulWidget {
  const ScannerPage({super.key});

  @override
  State<ScannerPage> createState() => _ScannerPageState();
}

class _ScannerPageState extends State<ScannerPage> with WidgetsBindingObserver {
  Future<void> _lifecycleWork = Future<void>.value();

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
    final bool shouldPause =
        state == AppLifecycleState.paused ||
        state == AppLifecycleState.inactive ||
        state == AppLifecycleState.hidden;
    final bool shouldResume = state == AppLifecycleState.resumed;
    if (!shouldPause && !shouldResume) return;

    // Lifecycle notifications commonly arrive as inactive -> hidden -> paused.
    // Serialize them so a quick resume cannot race a still-running camera pause.
    _lifecycleWork = _lifecycleWork
        .then((_) async {
          if (!mounted) return;
          final DetectionCubit cubit = context.read<DetectionCubit>();
          if (shouldPause) {
            await cubit.pauseForBackground();
          } else {
            await cubit.resumeFromBackground();
          }
        })
        .catchError((Object error, StackTrace stack) {
          debugPrint('[Lifecycle] camera transition failed: $error\n$stack');
        });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: BlocBuilder<DetectionCubit, DetectionState>(
        builder: (BuildContext context, DetectionState mlState) {
          return ScannerSurface(
            state: mlState,
            onTap: () => context.read<DetectionCubit>().toggle(),
          );
        },
      ),
    );
  }
}
