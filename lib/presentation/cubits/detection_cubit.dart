import 'dart:async';

import 'package:bigash/data/services/tts_service.dart';
import 'package:bigash/domain/currency/frame_decision.dart';
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

  /// Time-bounded majority voter over recent confident frames. The voting math
  /// (strict majority of a pruned window + deterministic tie-break) lives in
  /// the pure [MajorityVoter] so it can be unit-tested without plugins.
  final MajorityVoter _voter = MajorityVoter();

  /// Per-frame confidence / margin / OOD gates. Pure logic in [classifyFrame];
  /// see frame_decision.dart for what each threshold guards (including the
  /// out-of-distribution gate that stops a foreign note being read as money).
  static const DecisionThresholds _thresholds = DecisionThresholds();

  DateTime? _cooldownUntil;
  bool _canVibrate = false;
  bool _torchOn = false;
  /// Guards start()/stop() against re-entry — a rapid double-tap could
  /// otherwise run two camera start/stop cycles at once and end up using a
  /// disposed CameraController.
  bool _transitioning = false;
  /// Monotonic scan-session id, bumped on every start()/stop(). A delayed
  /// callback (e.g. the post-announcement cooldown timer that resumes after an
  /// awaited TTS call) compares the id it captured against this to detect that
  /// the session ended/restarted underneath it, and bails out so it cannot
  /// clobber a freshly-started scan.
  int _runId = 0;
  /// When the last spoken reposition hint played — throttles hint speech
  /// so it guides the user without talking over every frame.
  DateTime? _lastHintAt;
  /// When the last confident frame was voted — used to suppress hints
  /// while the app is actively detecting.
  DateTime? _lastVoteAt;
  /// When "العملة غير معروفة" was last announced — throttles it so the app
  /// states it once rather than nagging.
  DateTime? _lastOtherAnnounceAt;

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

    if (!_repository.isModelReady) {
      // Either the model file is missing, OR the model and labels.txt disagree
      // (a contract violation that would otherwise silently mis-map every
      // denomination). Surface the specific spoken reason when we have one.
      final String? contractMsg = _repository.modelErrorMessage;
      emit(
        state.copyWith(
          isActive: false,
          phase: CurrencyDetectionPhase.idle,
          statusText: contractMsg ??
              'لم يُضف ملف النموذج. ضع ملف tflite في المجلد assets وأعد بناء التطبيق.',
        ),
      );
      // Fire-and-forget: awaiting would hold toggle()'s [_transitioning]
      // guard for the whole sentence, swallowing a retry tap meanwhile.
      unawaited(
        _ttsService.speak(
          contractMsg ??
              'لم يُضف نموذج التعرف. أضف ملف النموذج إلى مجلد الأصول ثم أعد تشغيل التطبيق.',
        ),
      );
      return;
    }

    _voter.clear();
    _runId++;
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
      // Fire-and-forget: awaiting would hold toggle()'s [_transitioning]
      // guard for the whole sentence, swallowing the retry tap it invites.
      unawaited(
        _ttsService.speak(
          'تعذّر تشغيل الكاميرا. تأكّد من السماح للتطبيق باستخدام الكاميرا، '
          'ثم اضغط على الشاشة لإعادة المحاولة.',
        ),
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

  /// A distinct confirmation buzz fired on EVERY announcement, independent of
  /// TTS. The spoken denomination is the safety-critical output, but on a device
  /// whose Arabic voice is missing or fails silently, `speak()` returns with no
  /// audio and without throwing — so a blind user would otherwise get no
  /// confirmation that a result fired. This haptic guarantees a non-audio cue.
  /// The two patterns differ on purpose: a recognized bill is two firm pulses,
  /// an unrecognized one is a single long pulse, so the user can tell them apart
  /// even when they cannot hear the spoken phrase.
  void _confirmVibrate({required bool recognized}) {
    if (!_canVibrate) {
      return;
    }
    unawaited(Vibration.cancel());
    unawaited(
      Vibration.vibrate(
        pattern: recognized
            ? const <int>[0, 200, 120, 200]
            : const <int>[0, 500],
      ),
    );
  }

  Future<void> _setTorch(bool on) async {
    if (_torchOn == on) {
      return;
    }
    // Only update the cached state if the hardware actually accepted the
    // change, so a failed toggle can't desync the cache (torch stuck on, or the
    // low-light assist believing it is already on and never retrying).
    final bool ok = await _repository.setTorchEnabled(on);
    if (ok) {
      _torchOn = on;
    }
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
        .take(3)
        .map((e) => '${e.key} ${(e.value * 100).toStringAsFixed(0)}%')
        .join('  |  ');
  }

  void _onCurrencyEntity(CurrencyEntity entity) {
    if (!state.isActive || _inCooldown) {
      return;
    }

    final String debugText = _buildDebugText(entity);
    // Group the raw sub-variant top-K (y100_new/y100_old, ...) into parent
    // denominations BEFORE any gate, so a worn bill splitting its mass across
    // its two designs isn't read as a near-tie and suppressed. classifyFrame
    // then applies the confidence/margin/OOD gates — all pure, all tested.
    final List<MapEntry<String, double>> grouped = groupTopK(
      entity.topK.isNotEmpty
          ? entity.topK
          : <MapEntry<String, double>>[
              MapEntry<String, double>(entity.label, entity.confidence),
            ],
    );
    final FrameAssessment assessment = classifyFrame(grouped, _thresholds);
    final DateTime now = DateTime.now();

    switch (assessment.verdict) {
      case FrameVerdict.currency:
        unawaited(_setTorch(false));
        _lastVoteAt = now;
        _voter.add(assessment.key, now);
        final String? winner = _voter.winner(now);
        if (winner != null && winner == assessment.key) {
          _announceCurrency(winner, debugText, now);
          return;
        }
        emit(
          state.copyWith(
            phase: CurrencyDetectionPhase.scanning,
            statusText: 'جاري المسح',
            debugText: debugText,
          ),
        );
        return;

      case FrameVerdict.unknown:
        unawaited(_setTorch(false));
        _voter.add('other', now);
        if (_voter.winner(now) == 'other') {
          // Decisively "not a Yemeni bill". Say so once, then stay quiet for a
          // while so the app does not nag.
          _voter.clear();
          final bool tellUser = _lastOtherAnnounceAt == null ||
              now.difference(_lastOtherAnnounceAt!) >
                  const Duration(seconds: 12);
          if (tellUser) {
            _lastOtherAnnounceAt = now;
            _cooldownUntil = now.add(_speechCooldown);
            _stopPulsingVibration();
            _confirmVibrate(recognized: false);
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
        return;

      case FrameVerdict.uncertain:
        // Low confidence or ambiguous (incl. the OOD case where `other`
        // competes with the top Yemeni class). Guide the user — unless a
        // confident frame was just voted, in which case a hint would confuse.
        // "المسافة" confines the user to the near/far axis only (see the long
        // rationale that used to live here): no direction word, no side-to-side
        // motion, no "hold steady" clause that contradicts "change distance",
        // and no "اقترب" (which this device's TTS misreads).
        unawaited(_setTorch(true));
        const String hint = 'غيّر المسافة بين العملة والكاميرا ببطء';
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
  }

  /// Announces a confirmed denomination. Speaks the GROUPED winner (e.g. y100),
  /// not a single frame's sub-variant, so the spoken phrase always matches the
  /// voted result; if the grouped key is somehow unmapped it says "unknown"
  /// rather than reading a raw English token (e.g. "y100") aloud to a blind user.
  void _announceCurrency(String groupKey, String debugText, DateTime now) {
    final String mapped = arabicTtsForYemeniBill(groupKey);
    final bool recognized = mapped != groupKey;
    final String spoken = recognized ? mapped : 'العملة غير معروفة';
    _voter.clear();
    _cooldownUntil = now.add(_speechCooldown);
    _stopPulsingVibration();
    _confirmVibrate(recognized: recognized);
    unawaited(_setTorch(false));
    emit(
      state.copyWith(
        phase: CurrencyDetectionPhase.success,
        statusText: spoken,
        debugText: debugText,
      ),
    );
    unawaited(_finishSuccessAnnouncement(spoken));
  }

  /// True if a confident frame was voted within the last few seconds — the
  /// app is mid-detection, so a "reposition" hint would only confuse.
  bool _detectingRecently() =>
      _lastVoteAt != null &&
      DateTime.now().difference(_lastVoteAt!) < const Duration(seconds: 3);

  Future<void> _finishSuccessAnnouncement(String text) async {
    final int runId = _runId;
    await _ttsService.speak(text);
    // If the scan was stopped or restarted while we awaited the announcement,
    // this call belongs to a finished session — leave the current session's
    // cooldown timer and state untouched.
    if (isClosed || runId != _runId) {
      return;
    }
    _cooldownTimer?.cancel();
    _cooldownTimer = Timer(_speechCooldown, () {
      if (isClosed || runId != _runId) {
        return;
      }
      _cooldownUntil = null;
      if (!state.isActive) {
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
    // Invalidate any in-flight success-announcement continuation and its
    // cooldown timer so they cannot fire while we are backgrounded — otherwise
    // a cooldown that was armed just before backgrounding buzzes the phone in
    // the user's pocket and emits a stale "scanning" state for a paused camera.
    // Bumping _runId makes the awaited-TTS continuation in
    // _finishSuccessAnnouncement bail instead of re-arming a fresh timer after
    // _ttsService.stop() unblocks it. Clearing _cooldownUntil also stops a
    // resume that lands mid-cooldown from silently dropping frames.
    _runId++;
    _cooldownTimer?.cancel();
    _cooldownTimer = null;
    _cooldownUntil = null;
    _stopPulsingVibration();
    unawaited(WakelockPlus.disable());
    await _repository.pauseCamera();
    // pauseCamera drives the hardware torch off directly (bypassing _setTorch),
    // so resync the cache — otherwise the low-light assist believes the torch is
    // still on after resume and the early-return in _setTorch never re-enables it.
    _torchOn = false;
    await _ttsService.stop();
  }

  Future<void> resumeFromBackground() async {
    if (!state.isActive) {
      return;
    }
    // Stale votes from before the background pause are no longer
    // meaningful — what the user was scanning might be totally different
    // now. Reset so we don't false-trigger on resume.
    _voter.clear();
    final bool resumed = await _repository.resumeCamera();
    if (!resumed) {
      // The capture session did not come back (common on iOS after a long
      // background, or if camera permission was revoked while away). Don't
      // leave a frozen "scanning" screen with no live frames — tear down and
      // tell the user out loud to tap to restart.
      await stop(announce: false);
      unawaited(
        _ttsService.speak(
          'تعذّر استئناف الكاميرا. اضغط على الشاشة لإعادة التشغيل.',
        ),
      );
      return;
    }
    _startPulsingVibration();
    unawaited(WakelockPlus.enable());
  }

  Future<void> stop({bool announce = true}) async {
    if (!state.isActive) {
      return;
    }
    _runId++;
    _stopPulsingVibration();
    unawaited(WakelockPlus.disable());
    _cooldownTimer?.cancel();
    _cooldownTimer = null;
    _cooldownUntil = null;
    await _currencySubscription?.cancel();
    _currencySubscription = null;
    _voter.clear();
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
    if (announce) {
      // Tell the blind user out loud that scanning has stopped — speak() first
      // cuts any announcement that was still playing. Fire-and-forget on
      // purpose: awaiting it would hold toggle()'s [_transitioning] guard for
      // the full ~4s the sentence takes to speak (TTS uses
      // awaitSpeakCompletion), so a quick second tap to restart would be
      // swallowed until the sentence finished. Not awaiting lets the next tap
      // start() immediately — start()'s own speak() cuts this sentence off.
      // Callers that will speak their own message (e.g. resume-failure) pass
      // announce:false to avoid two overlapping utterances.
      unawaited(
        _ttsService.speak('تم إيقاف التعرف. اضغط على الشاشة لإعادة التشغيل.'),
      );
    }
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
