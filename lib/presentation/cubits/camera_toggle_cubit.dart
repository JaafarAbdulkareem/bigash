import 'package:flutter_bloc/flutter_bloc.dart';

class CameraToggleCubit extends Cubit<bool> {
  CameraToggleCubit() : super(false);

  void turnOn() => emit(true);

  void turnOff() => emit(false);

  void toggle() => emit(!state);
}
