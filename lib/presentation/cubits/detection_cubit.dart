import 'dart:async';

import 'package:bigash/data/services/tts_service.dart';
import 'package:bigash/domain/currency/yemeni_currency_label_mapper.dart';
import 'package:bigash/domain/entities/currency_entity.dart';
import 'package:bigash/domain/repositories/currency_repository.dart';
import 'package:camera/camera.dart';
import 'package:equatable/equatable.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:vibration/vibration.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

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
  /// Guards start()/stop() against re-entry — a rapid double-tap could
  /// otherwise run two camera start/stop cycles at once and end up using a
  /// disposed CameraController.
  bool _transitioning = false;
  /// When the last spoken reposition hint played — throttles hint speech
  /// so it guides the user without talking over every frame.
  DateTime? _lastHintAt;
  /// When the last confident frame was voted — used to suppress hints
  /// while the app is actively detecting.
  DateTime? _lastVoteAt;
  /// When "العملة غير معروفة" was last announced — throttles it so the app
  /// states it once rather than nagging.
  DateTime? _lastOtherAnnounceAt;

  /// Noise floor for a currency-class frame to count as a vote. This model
  /// is trained with label smoothing + MixUp, so its softmax is very soft —
  /// a correct top class often sits around 0.35-0.55, not 0.9+. So the
  /// absolute value is NOT the real signal: the announced class is simply
  /// the highest-probability one, and what guards against a wrong
  /// announcement is [_voteMargin] (a clear lead over the runner-up) plus
  /// the voting window (it must stay the winner across several frames).
  static const double _currencyThreshold = 0.35;
  /// Noise floor for an "other" frame — slightly lower, since a foreign
  /// note's own "other" score is also soft (~0.3-0.5).
  static const double _otherThreshold = 0.28;
  /// The top class must beat the runner-up by at least this margin to count
  /// as a vote. This is the PRIMARY confidence gate for this soft model — it
  /// blocks a genuine near-tie (e.g. other 29% vs y50 20%) from voting.
  static const double _voteMargin = 0.10;
  /// Number of recent frames we remember for majority voting.
  static const int _voteWindowSize = 10;
  /// How many of those frames must agree to announce. 3 keeps confirmation
  /// quick while still rejecting a stray wrong frame.
  static const int _requiredMatches = 3;
  static const Duration _speechCooldown = Duration(seconds: 3);

  bool get _inCooldown =>
      _cooldownUntil != null && DateTime.now().isBefore(_cooldownUntil!);

  Future<void> start() async {
    if (state.isActive) {
      return;
    }

    // Tapping to start cuts off the intro guidance immediately and confirms
    // out loud that scanning has begun — speak() stops in-progress speech
    // first, so the intro is interrupted the moment the user taps.
    unawaited(_ttsService.speak('بدأ المسح. وجّه الكاميرا نحو العملة.'));

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
    _lastVoteAt = null;
    _lastOtherAnnounceAt = null;
    _cooldownTimer?.cancel();
    _canVibrate = await Vibration.hasVibrator();

    await _repository.startCamera();

    // If the camera could not open, say so out loud instead of leaving the
    // blind user on a silent screen that never detects anything.
    if (!_repository.isCameraReady) {
      await _repository.stopCamera();
      emit(
        state.copyWith(
          isActive: false,
          phase: CurrencyDetectionPhase.idle,
          statusText:
              'تعذّر تشغيل الكاميرا. تأكّد من الإذن ثم اضغط لإعادة المحاولة.',
          clearCamera: true,
        ),
      );
      await _ttsService.speak(
        'تعذّر تشغيل الكاميرا. تأكّد من السماح للتطبيق باستخدام الكاميرا، '
        'ثم اضغط على الشاشة لإعادة المحاولة.',
      );
      return;
    }

    await _repository.setTorchEnabled(false);
    _torchOn = false;

    // Hold the screen awake while scanning: the user keeps the phone still
    // on a bill and would not notice the display timing out — which would
    // background the app and silently stop detection.
    unawaited(WakelockPlus.enable());

    emit(
      state.copyWith(
        isActive: true,
        phase: CurrencyDetectionPhase.scanning,
        statusText: 'جاري المسح',
        controller: _repository.cameraController,
      ),
    );

    _startPulsingVibration();
    _lastHintAt = DateTime.now();

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

  /// Spoken when the scanner screen first opens — the screen is otherwise
  /// silent, so a blind user needs to be told what to do.
  Future<void> announceIntro() async {
    if (state.isActive) {
      return;
    }
    await _ttsService.speak(
      'شاشة التعرف على العملة جاهزة. اضغط في أي مكان على الشاشة لبدء المسح.',
    );
  }

  /// Speaks a reposition hint, throttled to once every 12 seconds so it
  /// guides the user calmly instead of repeating annoyingly.
  void _maybeSpeakHint(String hint) {
    final DateTime now = DateTime.now();
    if (_lastHintAt != null &&
        now.difference(_lastHintAt!) < const Duration(seconds: 12)) {
      return;
    }
    _lastHintAt = now;
    unawaited(_ttsService.speak(hint));
  }

  String _buildDebugText(CurrencyEntity entity) {
    if (entity.topK.isEmpty) {
      return '${entity.label} ${(entity.confidence * 100).toStringAsFixed(0)}%';
    }
    return entity.topK
        .map((e) => '${e.key} ${(e.value * 100).toStringAsFixed(0)}%')
        .join('  |  ');
  }

  /// True if the top prediction beats the runner-up by [_voteMargin].
  /// Skips voting on ambiguous frames where two classes are nearly tied
  /// (e.g. y200=42%, y500=39%) — those frames silently turn the torch on
  /// and let the user reposition the bill.
  bool _hasMargin(CurrencyEntity entity) {
    if (entity.topK.length < 2) {
      return true; // not enough info; trust the top class
    }
    final double top = entity.topK[0].value;
    final double second = entity.topK[1].value;
    return (top - second) >= _voteMargin;
  }

  void _onCurrencyEntity(CurrencyEntity entity) {
    if (!state.isActive || _inCooldown) {
      return;
    }

    final String debugText = _buildDebugText(entity);
    final String key = normalizeYemeniClassifierLabel(entity.label);
    // Vote on the parent denomination, not the design variant, so frames
    // that flip between y100_new and y100_old on the same bill still
    // accumulate toward a single announcement.
    final String voteKey = denominationGroupForVoting(key);
    final bool hasMargin = _hasMargin(entity);

    if (isIgnoredYemeniCurrencyLabel(key)) {
      if (entity.confidence >= _otherThreshold && hasMargin) {
        unawaited(_setTorch(false));
        _pushVote(voteKey);
        final String? winner = _majorityWinner();
        if (winner != null && winner == voteKey) {
          // Decisively "not a Yemeni bill". Say so once, then stay quiet
          // for a while so the app does not nag.
          _voteWindow.clear();
          final DateTime now = DateTime.now();
          final bool tellUser = _lastOtherAnnounceAt == null ||
              now.difference(_lastOtherAnnounceAt!) >
                  const Duration(seconds: 12);
          if (tellUser) {
            _lastOtherAnnounceAt = now;
            _cooldownUntil = now.add(_speechCooldown);
            _stopPulsingVibration();
            emit(
              state.copyWith(
                phase: CurrencyDetectionPhase.success,
                statusText: 'العملة غير معروفة',
                debugText: debugText,
              ),
            );
            unawaited(_finishSuccessAnnouncement('العملة غير معروفة'));
            return;
          }
        }
        emit(
          state.copyWith(
            phase: CurrencyDetectionPhase.scanning,
            statusText: 'جاري المسح',
            debugText: debugText,
          ),
        );
      } else {
        // Genuinely unsure. Guide the user — unless a confident frame was
        // just voted, in which case a hint would only confuse.
        unawaited(_setTorch(true));
        const String hint = 'حاول تقريب العملة وثبّتها أمام الكاميرا';
        emit(
          state.copyWith(
            phase: CurrencyDetectionPhase.lowLight,
            statusText: hint,
            debugText: debugText,
          ),
        );
        if (!_detectingRecently()) {
          _maybeSpeakHint(hint);
        }
      }
      return;
    }

    if (entity.confidence <= _currencyThreshold || !hasMargin) {
      // Currency suspected but not confident. Could be: low light, bad
      // angle, hand shaking, or the bill is partially out of frame. Torch
      // on to help; speak the hint only if we are not already mid-detection
      // (a confident frame was just voted) — otherwise it just confuses.
      unawaited(_setTorch(true));
      const String hint = 'ثبّت يدك واقترب من العملة';
      emit(
        state.copyWith(
          phase: CurrencyDetectionPhase.lowLight,
          statusText: hint,
          debugText: debugText,
        ),
      );
      if (!_detectingRecently()) {
        _maybeSpeakHint(hint);
      }
      return;
    }

    unawaited(_setTorch(false));
    _pushVote(voteKey);

    final String? winner = _majorityWinner();
    if (winner != null && winner == voteKey) {
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

    emit(
      state.copyWith(
        phase: CurrencyDetectionPhase.scanning,
        statusText: 'جاري المسح',
        debugText: debugText,
      ),
    );
  }

  void _pushVote(String label) {
    _lastVoteAt = DateTime.now();
    _voteWindow.add(label);
    if (_voteWindow.length > _voteWindowSize) {
      _voteWindow.removeAt(0);
    }
  }

  /// True if a confident frame was voted within the last few seconds — the
  /// app is mid-detection, so a "reposition" hint would only confuse.
  bool _detectingRecently() =>
      _lastVoteAt != null &&
      DateTime.now().difference(_lastVoteAt!) < const Duration(seconds: 3);

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

  /// Pauses scanning when the app goes to the background. Keeps the camera
  /// controller alive so we can resume quickly — disposing+reopening on
  /// every backgrounding event causes visible delay and battery churn.
  Future<void> pauseForBackground() async {
    if (!state.isActive) {
      return;
    }
    _stopPulsingVibration();
    unawaited(WakelockPlus.disable());
    await _repository.pauseCamera();
    await _ttsService.stop();
  }

  Future<void> resumeFromBackground() async {
    if (!state.isActive) {
      return;
    }
    // Stale votes from before the background pause are no longer
    // meaningful — what the user was scanning might be totally different
    // now. Reset so we don't false-trigger on resume.
    _voteWindow.clear();
    await _repository.resumeCamera();
    _startPulsingVibration();
    unawaited(WakelockPlus.enable());
  }

  Future<void> stop() async {
    if (!state.isActive) {
      return;
    }
    _stopPulsingVibration();
    unawaited(WakelockPlus.disable());
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
    // Tell the blind user out loud that scanning has stopped — speak()
    // first cuts any announcement that was still playing.
    await _ttsService.speak('تم إيقاف التعرف. اضغط على الشاشة لإعادة التشغيل.');
  }

  Future<void> toggle() async {
    // Ignore taps that arrive while a previous start/stop is still running —
    // re-entry races two camera lifecycles and crashes on a disposed
    // CameraController.
    if (_transitioning) {
      return;
    }
    _transitioning = true;
    try {
      if (state.isActive) {
        await stop();
      } else {
        await start();
      }
    } finally {
      _transitioning = false;
    }
  }

  @override
  Future<void> close() async {
    _stopPulsingVibration();
    unawaited(WakelockPlus.disable());
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
