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
  final String debugText;
  final CameraController? controller;
  final CurrencyDetectionPhase phase;

  const DetectionState({
    required this.isActive,
    required this.isInitialized,
    required this.statusText,
    required this.phase,
    this.debugText = '',
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
    String? debugText,
    CameraController? controller,
    bool clearCamera = false,
    CurrencyDetectionPhase? phase,
  }) {
    return DetectionState(
      isActive: isActive ?? this.isActive,
      isInitialized: isInitialized ?? this.isInitialized,
      statusText: statusText ?? this.statusText,
      debugText: debugText ?? this.debugText,
      controller: clearCamera ? null : (controller ?? this.controller),
      phase: phase ?? this.phase,
    );
  }

  @override
  List<Object?> get props => <Object?>[
        isActive,
        isInitialized,
        statusText,
        debugText,
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

  /// Rolling window of recent high-confidence label votes for majority voting.
  /// This is more forgiving than strict consecutive matching for users who
  /// cannot hold the phone perfectly still (blind users, hand tremors).
  final List<String> _voteWindow = <String>[];
  DateTime? _cooldownUntil;
  bool _canVibrate = false;
  bool _torchOn = false;

  /// Confidence floor: frames below this are treated as "low light / uncertain".
  static const double _confidenceThreshold = 0.75;
  /// Number of frames kept in the voting window.
  static const int _voteWindowSize = 6;
  /// Required matching votes inside the window to confirm a bill.
  static const int _requiredMatches = 4;
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

    _voteWindow.clear();
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

  String _buildDebugText(CurrencyEntity entity) {
    if (entity.topK.isEmpty) {
      return '${entity.label} ${(entity.confidence * 100).toStringAsFixed(0)}%';
    }
    return entity.topK
        .map((e) => '${e.key} ${(e.value * 100).toStringAsFixed(0)}%')
        .join('  |  ');
  }

  void _onCurrencyEntity(CurrencyEntity entity) {
    if (!state.isActive || _inCooldown) {
      return;
    }

    final String debugText = _buildDebugText(entity);
    final String key = normalizeYemeniClassifierLabel(entity.label);

    if (isIgnoredYemeniCurrencyLabel(key)) {
      // Non-currency frame — push it into the window too, so a bill must
      // *outvote* the background to be announced.
      _pushVote(key);
      unawaited(_setTorch(false));
      emit(
        state.copyWith(
          phase: CurrencyDetectionPhase.scanning,
          statusText: 'جاري المسح',
          debugText: debugText,
        ),
      );
      return;
    }

    if (entity.confidence <= _confidenceThreshold) {
      // Low-confidence frames don't vote; turn on torch to help.
      unawaited(_setTorch(true));
      emit(
        state.copyWith(
          phase: CurrencyDetectionPhase.lowLight,
          statusText: 'إضاءة منخفضة — تم تشغيل الفلاش',
          debugText: debugText,
        ),
      );
      return;
    }

    unawaited(_setTorch(false));
    _pushVote(key);

    final String? winner = _majorityWinner();
    if (winner != null && winner == key) {
      final String spoken = entity.displayName;
      _voteWindow.clear();
      _cooldownUntil = DateTime.now().add(_speechCooldown);
      _stopPulsingVibration();
      unawaited(_setTorch(false));
      emit(
        state.copyWith(
          phase: CurrencyDetectionPhase.success,
          statusText: spoken,
          debugText: debugText,
        ),
      );
      unawaited(_finishSuccessAnnouncement(spoken));
      return;
    }

    final int matches = _voteWindow.where((v) => v == key).length;
    emit(
      state.copyWith(
        phase: CurrencyDetectionPhase.scanning,
        statusText: 'جاري التحقق $matches من $_requiredMatches',
        debugText: debugText,
      ),
    );
  }

  void _pushVote(String label) {
    _voteWindow.add(label);
    if (_voteWindow.length > _voteWindowSize) {
      _voteWindow.removeAt(0);
    }
  }

  /// Returns the label that appears at least [_requiredMatches] times in the
  /// current voting window, or null if no label reaches the threshold.
  String? _majorityWinner() {
    if (_voteWindow.length < _requiredMatches) {
      return null;
    }
    final Map<String, int> counts = <String, int>{};
    for (final String label in _voteWindow) {
      counts[label] = (counts[label] ?? 0) + 1;
    }
    String? best;
    int bestCount = 0;
    counts.forEach((String label, int count) {
      if (count > bestCount) {
        bestCount = count;
        best = label;
      }
    });
    if (bestCount >= _requiredMatches) {
      return best;
    }
    return null;
  }

  Future<void> cycleNormalization() async {
    _repository.cycleNormalization();
    _voteWindow.clear();
    await _ttsService.speak('تم تغيير وضع المعالجة: ${_repository.diagnostics}');
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
    _voteWindow.clear();
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
