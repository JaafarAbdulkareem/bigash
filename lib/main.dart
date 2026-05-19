import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:bigash/core/di/injection_container.dart' as di;
import 'package:bigash/data/services/tts_service.dart';
import 'package:bigash/presentation/cubits/camera_toggle_cubit.dart';
import 'package:bigash/presentation/cubits/detection_cubit.dart';
import 'package:bigash/presentation/pages/splash_page.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Lock to portrait. Blind users hold the phone upright; rotating would
  // confuse the camera orientation logic and produce wrong predictions.
  await SystemChrome.setPreferredOrientations(<DeviceOrientation>[
    DeviceOrientation.portraitUp,
  ]);

  // Keep the screen on while scanning — a blind user can't tell that the
  // display has dimmed before timeout and might think the app stopped.
  // (Implemented via persistent overlay + scanner-page-level wake handling.)

  di.init();
  runApp(const BigashApp());
}

class BigashApp extends StatelessWidget {
  const BigashApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiBlocProvider(
      providers: [
        BlocProvider<CameraToggleCubit>(
          create: (_) => di.sl<CameraToggleCubit>(),
        ),
        BlocProvider<DetectionCubit>(
          create: (_) => di.sl<DetectionCubit>(),
        ),
      ],
      child: MaterialApp(
        title: 'Bigash',
        debugShowCheckedModeBanner: false,
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
