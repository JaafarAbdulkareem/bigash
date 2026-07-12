import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:bigash/core/di/injection_container.dart' as di;
import 'package:bigash/data/services/tts_service.dart';
import 'package:bigash/presentation/cubits/detection_cubit.dart';
import 'package:bigash/presentation/pages/splash_page.dart';

void main() {
  // Run inside a guarded zone so an otherwise-unhandled async error does
  // not hard-crash the app — a crash leaves a blind user with no feedback
  // at all. Errors are logged; the UI keeps running.
  runZonedGuarded<Future<void>>(
    () async {
      WidgetsFlutterBinding.ensureInitialized();

      FlutterError.onError = (FlutterErrorDetails details) {
        FlutterError.presentError(details);
        debugPrint('[FlutterError] ${details.exception}');
      };

      // Lock to portrait. Blind users hold the phone upright; rotating would
      // confuse the camera orientation logic and produce wrong predictions.
      await SystemChrome.setPreferredOrientations(<DeviceOrientation>[
        DeviceOrientation.portraitUp,
      ]);

      di.init();
      runApp(const BigashApp());
    },
    (Object error, StackTrace stack) {
      debugPrint('[Uncaught] $error\n$stack');
    },
  );
}

class BigashApp extends StatelessWidget {
  const BigashApp({super.key});

  @override
  Widget build(BuildContext context) {
    return BlocProvider<DetectionCubit>(
      create: (_) => di.sl<DetectionCubit>(),
      child: MaterialApp(
        title: 'بقش',
        debugShowCheckedModeBanner: false,
        builder: (BuildContext context, Widget? child) {
          return Directionality(
            textDirection: TextDirection.rtl,
            child: child ?? const SizedBox.shrink(),
          );
        },
        theme: ThemeData(
          brightness: Brightness.dark,
          primaryColor: Colors.black,
          scaffoldBackgroundColor: Colors.black,
        ),
        home: SplashPage(ttsService: di.sl<TTSService>()),
      ),
    );
  }
}
