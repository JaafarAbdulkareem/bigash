import 'package:get_it/get_it.dart';
import 'package:bigash/data/datasources/camera_service.dart';
import 'package:bigash/data/repositories/currency_repository_impl.dart';
import 'package:bigash/data/services/tflite_service.dart';
import 'package:bigash/data/services/tts_service.dart';
import 'package:bigash/domain/repositories/currency_repository.dart';
import 'package:bigash/presentation/cubits/camera_toggle_cubit.dart';
import 'package:bigash/presentation/cubits/detection_cubit.dart';
import 'package:flutter_tts/flutter_tts.dart';

final sl = GetIt.instance;

void init() {
  sl.registerLazySingleton<CameraService>(() => CameraService());
  sl.registerLazySingleton<TFLiteService>(() => TFLiteService());
  sl.registerLazySingleton<FlutterTts>(() => FlutterTts());
  sl.registerLazySingleton<TTSService>(() => TTSService(sl()));

  sl.registerLazySingleton<CurrencyRepository>(() => CurrencyRepositoryImpl(
    cameraService: sl(),
    tfliteService: sl(),
  ));

  sl.registerFactory<CameraToggleCubit>(() => CameraToggleCubit());
  sl.registerFactory<DetectionCubit>(
    () => DetectionCubit(
      repository: sl(),
      ttsService: sl(),
    ),
  );
}
