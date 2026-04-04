import 'dart:async';

import 'package:bigash/data/services/tts_service.dart';
import 'package:bigash/domain/entities/currency_entity.dart';
import 'package:bigash/domain/repositories/currency_repository.dart';
import 'package:bigash/domain/usecases/recognize_currency_usecase.dart';
import 'package:camera/camera.dart';
import 'package:equatable/equatable.dart';
import 'package:flutter/services.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

class DetectionState extends Equatable {
  final bool isActive;
  final bool isInitialized;
  final String statusText;
  final CameraController? controller;
  final DetectionVerificationState verification;

  const DetectionState({
    required this.isActive,
    required this.isInitialized,
    required this.statusText,
    required this.verification,
    this.controller,
  });

  factory DetectionState.initial() {
    return const DetectionState(
      isActive: false,
      isInitialized: false,
      statusText: 'اضغط على الشاشة لبدء التعرف',
      verification: DetectionVerificationState.inconsistent,
    );
  }

  DetectionState copyWith({
    bool? isActive,
    bool? isInitialized,
    String? statusText,
    CameraController? controller,
    DetectionVerificationState? verification,
  }) {
    return DetectionState(
      isActive: isActive ?? this.isActive,
      isInitialized: isInitialized ?? this.isInitialized,
      statusText: statusText ?? this.statusText,
      controller: controller ?? this.controller,
      verification: verification ?? this.verification,
    );
  }

  @override
  List<Object?> get props => <Object?>[
        isActive,
        isInitialized,
        statusText,
        controller,
        verification,
      ];
}

/// Strict verification guard: 5 identical high-confidence frames, cooldown, haptics only when unsure.
class DetectionCubit extends Cubit<DetectionState> {
  DetectionCubit({
    required CurrencyRepository repository,
    required TTSService ttsService,
    required RecognizeCurrencyUseCase recognizeCurrencyUseCase,
  })  : _repository = repository,
        _ttsService = ttsService,
        _recognizeCurrencyUseCase = recognizeCurrencyUseCase,
        super(DetectionState.initial());

  final CurrencyRepository _repository;
  final TTSService _ttsService;
  final RecognizeCurrencyUseCase _recognizeCurrencyUseCase;

  StreamSubscription<CurrencyEntity>? _currencySubscription;

  final List<DetectionFrame> _resultsBuffer = <DetectionFrame>[];
  DateTime? _cooldownUntil;
  DateTime _lastHapticAt =
      DateTime.now().subtract(const Duration(seconds: 1));

  static const Duration _speechCooldown = Duration(seconds: 3);
  static const Duration _hapticMinGap = Duration(milliseconds: 400);

  bool get _inCooldown =>
      _cooldownUntil != null && DateTime.now().isBefore(_cooldownUntil!);

  Future<void> start() async {
    if (state.isActive) {
      return;
    }

    if (!state.isInitialized) {
      await _repository.initialize();
      emit(state.copyWith(isInitialized: true));
    }

    if (!_repository.isModelLoaded) {
      emit(
        state.copyWith(
          isActive: false,
          verification: DetectionVerificationState.inconsistent,
          statusText:
              'لم يُضف ملف النموذج. ضع ملف tflite في المجلد assets وأعد بناء التطبيق.',
        ),
      );
      await _ttsService.speak(
        'لم يُضف نموذج التعرف. أضف ملف النموذج إلى مجلد الأصول ثم أعد تشغيل التطبيق.',
      );
      return;
    }

    _resultsBuffer.clear();
    _cooldownUntil = null;
    await _repository.startCamera();

    emit(
      state.copyWith(
        isActive: true,
        verification: DetectionVerificationState.inconsistent,
        statusText: 'جاري التعرف...',
        controller: _repository.cameraController,
      ),
    );

    await _currencySubscription?.cancel();
    _currencySubscription =
        _repository.getCurrencyStream().listen(_onCurrencyEntity);
  }

  void _onCurrencyEntity(CurrencyEntity entity) {
    if (!state.isActive) {
      return;
    }

    if (_inCooldown) {
      return;
    }

    final StrictVerificationResult result = _recognizeCurrencyUseCase.applyFrame(
      buffer: List<DetectionFrame>.from(_resultsBuffer),
      entity: entity,
    );

    _resultsBuffer
      ..clear()
      ..addAll(result.newBuffer);

    emit(
      state.copyWith(
        verification: result.verificationState,
        statusText: result.statusHint ??
            _defaultStatusFor(result.verificationState),
      ),
    );

    if (result.shouldVibrate) {
      _maybeHapticFeedback();
    }

    if (result.speakText != null && result.speakText!.isNotEmpty) {
      _cooldownUntil = DateTime.now().add(_speechCooldown);
      final String said = result.speakText!;
      unawaited(_finishStableAnnouncement(said));
    }
  }

  String _defaultStatusFor(DetectionVerificationState v) {
    switch (v) {
      case DetectionVerificationState.stableMatch:
        return 'تم التأكد';
      case DetectionVerificationState.inconsistent:
        return 'جاري التحقق من الاستقرار';
      case DetectionVerificationState.lowConfidence:
        return 'لم يُثبت الإطار بعد';
    }
  }

  Future<void> _finishStableAnnouncement(String text) async {
    await _ttsService.speak(text);
    if (!isClosed && state.isActive) {
      emit(
        state.copyWith(
          verification: DetectionVerificationState.stableMatch,
          statusText: text,
        ),
      );
    }
  }

  void _maybeHapticFeedback() {
    final DateTime now = DateTime.now();
    if (now.difference(_lastHapticAt) < _hapticMinGap) {
      return;
    }
    _lastHapticAt = now;
    HapticFeedback.mediumImpact();
  }

  Future<void> stop() async {
    if (!state.isActive) {
      return;
    }
    await _currencySubscription?.cancel();
    _currencySubscription = null;
    _resultsBuffer.clear();
    _cooldownUntil = null;
    await _repository.stopCamera();
    await _ttsService.stop();
    emit(
      state.copyWith(
        isActive: false,
        verification: DetectionVerificationState.inconsistent,
        statusText: 'متوقف - اضغط على الشاشة للتشغيل',
      ),
    );
  }

  Future<void> toggle() async {
    if (state.isActive) {
      await stop();
    } else {
      await start();
    }
  }

  @override
  Future<void> close() async {
    await _currencySubscription?.cancel();
    _repository.dispose();
    return super.close();
  }
}
