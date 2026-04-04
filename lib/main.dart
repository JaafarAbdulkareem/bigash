import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:bigash/core/di/injection_container.dart' as di;
import 'package:bigash/data/services/tts_service.dart';
import 'package:bigash/presentation/cubits/camera_toggle_cubit.dart';
import 'package:bigash/presentation/cubits/detection_cubit.dart';
import 'package:bigash/presentation/pages/splash_page.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
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
