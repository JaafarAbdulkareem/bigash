import 'dart:async';

import 'package:bigash/data/services/tts_service.dart';
import 'package:bigash/domain/currency/yemeni_currency_label_mapper.dart';
import 'package:bigash/domain/entities/currency_entity.dart';
import 'package:bigash/domain/repositories/currency_repository.dart';
import 'package:camera/camera.dart';
import 'package:equatable/equatable.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:vibration/vibration.dart';

/// High-level UI / accessibility phase for currency scanning.
enum CurrencyDetectionPhase {
  idle,
  /// Scanning with pulsing vibration (torch follows confidence rules).
  scanning,
  /// Low confidence: flashlight on, vibration continues.
  lowLight,
  /// Just identified a bill; vibration stopped, TTS will play / playing.
  success,
}

class DetectionState extends Equatable {
  final bool isActive;
  final bool isInitialized;
  final String statusText;
  final CameraController? controller;
  final CurrencyDetectionPhase phase;

  const DetectionState({
    required this.isActive,
    required this.isInitialized,
    required this.statusText,
    required this.phase,
    this.controller,
  });

  factory DetectionState.initial() {
    return const DetectionState(
      isActive: false,
      isInitialized: false,
      statusText: 'اضغط على الشاشة لبدء التعرف',
      phase: CurrencyDetectionPhase.idle,
    );
  }

  DetectionState copyWith({
    bool? isActive,
    bool? isInitialized,
    String? statusText,
    CameraController? controller,
    bool clearCamera = false,
    CurrencyDetectionPhase? phase,
  }) {
    return DetectionState(
      isActive: isActive ?? this.isActive,
      isInitialized: isInitialized ?? this.isInitialized,
      statusText: statusText ?? this.statusText,
      controller: clearCamera ? null : (controller ?? this.controller),
      phase: phase ?? this.phase,
    );
  }

  @override
  List<Object?> get props => <Object?>[
        isActive,
        isInitialized,
        statusText,
        controller,
        phase,
      ];
}

class DetectionCubit extends Cubit<DetectionState> {
  DetectionCubit({
    required CurrencyRepository repository,
    required TTSService ttsService,
  })  : _repository = repository,
        _ttsService = ttsService,
        super(DetectionState.initial());

  final CurrencyRepository _repository;
  final TTSService _ttsService;

  StreamSubscription<CurrencyEntity>? _currencySubscription;
  Timer? _vibrationPulseTimer;
  Timer? _cooldownTimer;

  final List<String> _consecutiveResults = <String>[];
  DateTime? _cooldownUntil;
  bool _canVibrate = false;
  bool _torchOn = false;

  static const double _confidenceThreshold = 0.85;
  static const int _requiredConsecutive = 5;
  static const Duration _speechCooldown = Duration(seconds: 3);

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
          phase: CurrencyDetectionPhase.idle,
          statusText:
              'لم يُضف ملف النموذج. ضع ملف tflite في المجلد assets وأعد بناء التطبيق.',
        ),
      );
      await _ttsService.speak(
        'لم يُضف نموذج التعرف. أضف ملف النموذج إلى مجلد الأصول ثم أعد تشغيل التطبيق.',
      );
      return;
    }

    _consecutiveResults.clear();
    _cooldownUntil = null;
    _cooldownTimer?.cancel();
    _canVibrate = await Vibration.hasVibrator();

    await _repository.startCamera();
    await _repository.setTorchEnabled(false);
    _torchOn = false;

    emit(
      state.copyWith(
        isActive: true,
        phase: CurrencyDetectionPhase.scanning,
        statusText: 'جاري المسح',
        controller: _repository.cameraController,
      ),
    );

    _startPulsingVibration();

    await _currencySubscription?.cancel();
    _currencySubscription =
        _repository.getCurrencyStream().listen(_onCurrencyEntity);
  }

  void _startPulsingVibration() {
    _vibrationPulseTimer?.cancel();
    if (!_canVibrate) {
      return;
    }
    _vibrationPulseTimer = Timer.periodic(const Duration(milliseconds: 650), (_) {
      if (!state.isActive || _inCooldown) {
        return;
      }
      if (state.phase == CurrencyDetectionPhase.success) {
        return;
      }
      unawaited(Vibration.vibrate(duration: 90));
    });
  }

  void _stopPulsingVibration() {
    _vibrationPulseTimer?.cancel();
    _vibrationPulseTimer = null;
    unawaited(Vibration.cancel());
  }

  Future<void> _setTorch(bool on) async {
    if (_torchOn == on) {
      return;
    }
    _torchOn = on;
    await _repository.setTorchEnabled(on);
  }

  void _onCurrencyEntity(CurrencyEntity entity) {
    if (!state.isActive || _inCooldown) {
      return;
    }

    final String key = normalizeYemeniClassifierLabel(entity.label);

    if (isIgnoredYemeniCurrencyLabel(key)) {
      _consecutiveResults.clear();
      unawaited(_setTorch(false));
      emit(
        state.copyWith(
          phase: CurrencyDetectionPhase.scanning,
          statusText: 'جاري المسح',
        ),
      );
      return;
    }

    if (entity.confidence <= _confidenceThreshold) {
      _consecutiveResults.clear();
      unawaited(_setTorch(true));
      emit(
        state.copyWith(
          phase: CurrencyDetectionPhase.lowLight,
          statusText: 'إضاءة منخفضة — تم تشغيل الفلاش',
        ),
      );
      return;
    }

    unawaited(_setTorch(false));

    if (_consecutiveResults.isEmpty || _consecutiveResults.last == key) {
      _consecutiveResults.add(key);
    } else {
      _consecutiveResults
        ..clear()
        ..add(key);
    }

    if (_consecutiveResults.length == _requiredConsecutive) {
      final String spoken = entity.displayName;
      _consecutiveResults.clear();
      _cooldownUntil = DateTime.now().add(_speechCooldown);
      _stopPulsingVibration();
      unawaited(_setTorch(false));
      emit(
        state.copyWith(
          phase: CurrencyDetectionPhase.success,
          statusText: spoken,
        ),
      );
      unawaited(_finishSuccessAnnouncement(spoken));
      return;
    }

    emit(
      state.copyWith(
        phase: CurrencyDetectionPhase.scanning,
        statusText:
            'جاري التحقق ${_consecutiveResults.length} من $_requiredConsecutive',
      ),
    );
  }

  Future<void> _finishSuccessAnnouncement(String text) async {
    await _ttsService.speak(text);
    _cooldownTimer?.cancel();
    _cooldownTimer = Timer(_speechCooldown, () {
      _cooldownUntil = null;
      if (isClosed || !state.isActive) {
        return;
      }
      _startPulsingVibration();
      emit(
        state.copyWith(
          phase: CurrencyDetectionPhase.scanning,
          statusText: 'جاري المسح',
        ),
      );
    });
  }

  Future<void> stop() async {
    if (!state.isActive) {
      return;
    }
    _stopPulsingVibration();
    _cooldownTimer?.cancel();
    _cooldownTimer = null;
    _cooldownUntil = null;
    await _currencySubscription?.cancel();
    _currencySubscription = null;
    _consecutiveResults.clear();
    await _setTorch(false);

    // Drop [CameraPreview] before disposing so buildPreview() never runs on a dead controller.
    emit(
      state.copyWith(
        isActive: false,
        phase: CurrencyDetectionPhase.idle,
        statusText: 'متوقف - اضغط على الشاشة للتشغيل',
        clearCamera: true,
      ),
    );
    await Future<void>.delayed(Duration.zero);

    await _repository.stopCamera();
    await _ttsService.stop();
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
    _stopPulsingVibration();
    _cooldownTimer?.cancel();
    await _currencySubscription?.cancel();
    if (!isClosed) {
      emit(
        state.copyWith(
          isActive: false,
          phase: CurrencyDetectionPhase.idle,
          clearCamera: true,
        ),
      );
    }
    await Future<void>.delayed(Duration.zero);
    await _repository.setTorchEnabled(false);
    _repository.dispose();
    return super.close();
  }
}
