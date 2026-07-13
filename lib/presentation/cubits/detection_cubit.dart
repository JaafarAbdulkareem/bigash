import 'dart:async';

import 'package:bigash/data/services/tts_service.dart';
import 'package:bigash/domain/currency/frame_decision.dart';
import 'package:bigash/domain/currency/frame_quality.dart';
import 'package:bigash/domain/currency/reposition_guidance.dart';
import 'package:bigash/domain/currency/yemeni_currency_label_mapper.dart';
import 'package:bigash/domain/entities/currency_entity.dart';
import 'package:bigash/domain/repositories/currency_repository.dart';
import 'package:camera/camera.dart';
import 'package:equatable/equatable.dart';
import 'package:flutter/foundation.dart';
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
  }) : _repository = repository,
       _ttsService = ttsService,
       super(DetectionState.initial());

  final CurrencyRepository _repository;
  final TTSService _ttsService;

  StreamSubscription<CurrencyEntity>? _currencySubscription;
  bool _scanPulseEnabled = false;
  DateTime? _lastScanPulseAt;
  Timer? _cooldownTimer;
  Timer? _pipelineWatchdogTimer;

  /// Time-bounded majority voter over recent confident frames. The voting math
  /// (strict majority of a pruned window + deterministic tie-break) lives in
  /// the pure [MajorityVoter] so it can be unit-tested without plugins.
  final MajorityVoter _voter = MajorityVoter();
  final RepositionGuidancePolicy _repositionGuidance =
      RepositionGuidancePolicy();

  /// Per-frame confidence / margin / OOD gates. Pure logic in [classifyFrame];
  /// see frame_decision.dart for what each threshold guards (including the
  /// out-of-distribution gate that stops a foreign note being read as money).
  static const DecisionThresholds _thresholds = DecisionThresholds();
  static const bool _showDiagnostics = bool.fromEnvironment(
    'BIGASH_DIAGNOSTICS',
    defaultValue: !kReleaseMode,
  );

  DateTime? _cooldownUntil;
  bool _canVibrate = false;
  bool _torchOn = false;
  bool _desiredTorchOn = false;
  Future<void> _torchTransition = Future<void>.value();
  bool _pausedForBackground = false;
  bool _appInBackground = false;
  bool _announcementInProgress = false;

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

  /// When "ليست عملة يمنية" was last announced — throttles it so the app
  /// states it once rather than nagging.
  DateTime? _lastOtherAnnounceAt;
  DateTime? _lastInferenceResultAt;

  static const Duration _speechCooldown = Duration(seconds: 3);
  static const String _notYemeniCurrencyMessage = 'ليست عملة يمنية';
  static const String _cannotReadMessage =
      'تعذّر قراءة العملة، وقد لا تكون عملة يمنية.';

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

    // Retry initialization after a transient model-load failure. Marking the
    // first attempt as initialized must not make every later retry a no-op.
    if (!state.isInitialized || !_repository.isModelReady) {
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
          statusText:
              contractMsg ??
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

    // Model initialization may finish after a lifecycle pause. Never open the
    // camera once the app has already moved to the background.
    if (_appInBackground) {
      return;
    }

    _voter.clear();
    _repositionGuidance.reset();
    _pausedForBackground = false;
    _runId++;
    _cooldownUntil = null;
    _announcementInProgress = false;
    _lastVoteAt = null;
    _lastOtherAnnounceAt = null;
    _cooldownTimer?.cancel();
    _canVibrate = await Vibration.hasVibrator();

    await _repository.startCamera();

    // Camera startup can also outlive the foreground session. Tear down a
    // controller that finished opening after the app was backgrounded.
    if (_appInBackground) {
      await _repository.stopCamera();
      return;
    }

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
    _desiredTorchOn = false;

    // The torch request above is another asynchronous boundary where a
    // lifecycle pause can arrive before the scanner becomes active.
    if (_appInBackground) {
      await _repository.stopCamera();
      return;
    }

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
    _currencySubscription = _repository.getCurrencyStream().listen(
      _onCurrencyEntity,
    );
    _startPipelineWatchdog();
  }

  void _startPipelineWatchdog() {
    _pipelineWatchdogTimer?.cancel();
    _lastInferenceResultAt = DateTime.now();
    final int runId = _runId;
    _pipelineWatchdogTimer = Timer.periodic(const Duration(seconds: 5), (_) {
      if (isClosed ||
          runId != _runId ||
          !state.isActive ||
          _appInBackground ||
          _pausedForBackground) {
        return;
      }
      final DateTime? last = _lastInferenceResultAt;
      if (last != null &&
          DateTime.now().difference(last) > const Duration(seconds: 30)) {
        unawaited(_handlePipelineStall(runId));
      }
    });
  }

  void _stopPipelineWatchdog() {
    _pipelineWatchdogTimer?.cancel();
    _pipelineWatchdogTimer = null;
    _lastInferenceResultAt = null;
  }

  Future<void> _handlePipelineStall(int runId) async {
    if (_transitioning || runId != _runId || !state.isActive) return;
    _transitioning = true;
    try {
      await stop(announce: false);
      if (isClosed) return;
      const String message =
          'توقّف تحليل صورة الكاميرا. اضغط على الشاشة لإعادة المحاولة.';
      emit(state.copyWith(statusText: message));
      unawaited(_ttsService.speak(message));
    } finally {
      _transitioning = false;
    }
  }

  void _startPulsingVibration() {
    _scanPulseEnabled = _canVibrate;
    _lastScanPulseAt = null;
  }

  void _stopPulsingVibration() {
    _scanPulseEnabled = false;
    _lastScanPulseAt = null;
    unawaited(Vibration.cancel());
  }

  /// Pulse only after a frame has finished inference. The previous independent
  /// 90ms/650ms timer physically shook the camera while it was exposing the
  /// next frame. A short post-result pulse ends before the 500ms inference
  /// throttle accepts another frame, preserving tactile feedback without
  /// manufacturing motion blur ourselves.
  void _maybePulseAfterProcessedFrame(DateTime now) {
    if (!_scanPulseEnabled ||
        !state.isActive ||
        _inCooldown ||
        state.phase == CurrencyDetectionPhase.success) {
      return;
    }
    if (_lastScanPulseAt != null &&
        now.difference(_lastScanPulseAt!) < const Duration(seconds: 2)) {
      return;
    }
    _lastScanPulseAt = now;
    unawaited(Vibration.vibrate(duration: 35));
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

  Future<void> _setTorch(bool on) {
    _desiredTorchOn = on;
    // Frames can alternate confidence quickly. Serialize hardware requests so
    // an older `on` call cannot finish after a newer `off` call and leave the
    // torch enabled during success/background states.
    _torchTransition = _torchTransition
        .then((_) async {
          final bool target = _desiredTorchOn;
          if (_torchOn == target) return;
          final bool ok = await _repository.setTorchEnabled(target);
          if (ok) _torchOn = target;
        })
        .catchError((Object _) {
          // The repository normally converts hardware failures to false. Keep
          // this chain recoverable for alternate/test implementations.
        });
    return _torchTransition;
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
  void _maybeSpeakHint(String hint, {bool repositionHint = false}) {
    final DateTime now = DateTime.now();
    if (_lastHintAt != null &&
        now.difference(_lastHintAt!) < const Duration(seconds: 12)) {
      return;
    }
    String spoken = hint;
    if (repositionHint) {
      switch (_repositionGuidance.next()) {
        case RepositionGuidanceDecision.repeatHint:
          break;
        case RepositionGuidanceDecision.announceFailure:
          spoken = _cannotReadMessage;
          emit(state.copyWith(statusText: spoken));
          break;
        case RepositionGuidanceDecision.silent:
          return;
      }
    }
    _lastHintAt = now;
    unawaited(_ttsService.speak(spoken));
  }

  String _buildDebugText(CurrencyEntity entity) {
    if (!_showDiagnostics) return '';
    if (entity.topK.isEmpty) {
      return '${entity.label} ${(entity.confidence * 100).toStringAsFixed(0)}%';
    }
    return entity.topK
        .take(3)
        .map((e) => '${e.key} ${(e.value * 100).toStringAsFixed(0)}%')
        .join('  |  ');
  }

  void _onCurrencyEntity(CurrencyEntity entity) {
    _lastInferenceResultAt = DateTime.now();
    if (!state.isActive ||
        _appInBackground ||
        _pausedForBackground ||
        _announcementInProgress ||
        _inCooldown) {
      return;
    }

    final DateTime now = DateTime.now();
    if (entity.quality != CaptureQuality.usable) {
      _handlePoorFrame(entity, now);
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

    switch (assessment.verdict) {
      case FrameVerdict.currency:
        _repositionGuidance.reset();
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
        _maybePulseAfterProcessedFrame(now);
        return;

      case FrameVerdict.unknown:
        unawaited(_setTorch(false));
        _voter.add('other', now);
        if (_voter.winner(now) == 'other') {
          // Decisively "not a Yemeni bill". Say so once, then stay quiet for a
          // while so the app does not nag.
          _voter.clear();
          final bool tellUser =
              _lastOtherAnnounceAt == null ||
              now.difference(_lastOtherAnnounceAt!) >
                  const Duration(seconds: 12);
          if (tellUser) {
            _lastOtherAnnounceAt = now;
            _cooldownUntil = now.add(_speechCooldown);
            _announcementInProgress = true;
            _stopPulsingVibration();
            _confirmVibrate(recognized: false);
            emit(
              state.copyWith(
                phase: CurrencyDetectionPhase.success,
                statusText: _notYemeniCurrencyMessage,
                debugText: debugText,
              ),
            );
            unawaited(_finishSuccessAnnouncement(_notYemeniCurrencyMessage));
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
        _maybePulseAfterProcessedFrame(now);
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
          _maybeSpeakHint(hint, repositionHint: true);
        }
        _maybePulseAfterProcessedFrame(now);
        return;
    }
  }

  void _handlePoorFrame(CurrencyEntity entity, DateTime now) {
    final String hint;
    switch (entity.quality) {
      case CaptureQuality.tooDark:
        hint = 'الإضاءة ضعيفة. ثبّت الهاتف ووجّه الكاميرا نحو العملة';
        unawaited(_setTorch(true));
        break;
      case CaptureQuality.tooBright:
        hint = 'أبعد العملة عن الضوء المباشر وثبّت الهاتف';
        unawaited(_setTorch(false));
        break;
      case CaptureQuality.blurry:
        hint = 'ثبّت الهاتف أو اسنده، وقرّب العملة من الكاميرا ببطء';
        // Extra light shortens exposure time and reduces hand-motion blur.
        unawaited(_setTorch(entity.brightness < 130));
        break;
      case CaptureQuality.usable:
        return;
    }
    emit(
      state.copyWith(
        phase: CurrencyDetectionPhase.lowLight,
        statusText: hint,
        debugText: _showDiagnostics
            ? 'ضوء ${entity.brightness.toStringAsFixed(0)} | '
                  'حدة ${entity.sharpness.toStringAsFixed(1)}'
            : '',
      ),
    );
    if (!_detectingRecently()) {
      _maybeSpeakHint(
        hint,
        repositionHint: entity.quality == CaptureQuality.blurry,
      );
    }
    _maybePulseAfterProcessedFrame(now);
  }

  /// Announces a confirmed denomination. Speaks the GROUPED winner (e.g. y100),
  /// not a single frame's sub-variant, so the spoken phrase always matches the
  /// voted result; if the grouped key is somehow unmapped it says "unknown"
  /// rather than reading a raw English token (e.g. "y100") aloud to a blind user.
  void _announceCurrency(String groupKey, String debugText, DateTime now) {
    final String mapped = arabicTtsForYemeniBill(groupKey);
    final bool recognized = mapped != groupKey;
    final String spoken = recognized ? mapped : _notYemeniCurrencyMessage;
    _voter.clear();
    _repositionGuidance.reset();
    _cooldownUntil = now.add(_speechCooldown);
    _announcementInProgress = true;
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
    _announcementInProgress = false;
    _cooldownUntil = DateTime.now().add(_speechCooldown);
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
    _appInBackground = true;
    if (!state.isActive || _pausedForBackground) {
      await _ttsService.stop();
      return;
    }
    _pausedForBackground = true;
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
    _announcementInProgress = false;
    _stopPulsingVibration();
    _stopPipelineWatchdog();
    unawaited(WakelockPlus.disable());
    await _setTorch(false);
    await _repository.pauseCamera();
    // pauseCamera drives the hardware torch off directly (bypassing _setTorch),
    // so resync the cache — otherwise the low-light assist believes the torch is
    // still on after resume and the early-return in _setTorch never re-enables it.
    _torchOn = false;
    _desiredTorchOn = false;
    await _ttsService.stop();
  }

  Future<void> resumeFromBackground() async {
    _appInBackground = false;
    if (!state.isActive || !_pausedForBackground) {
      return;
    }
    // Stale votes from before the background pause are no longer
    // meaningful — what the user was scanning might be totally different
    // now. Reset so we don't false-trigger on resume.
    _voter.clear();
    _repositionGuidance.reset();
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
    if (isClosed || !state.isActive) {
      return;
    }
    emit(
      state.copyWith(
        phase: CurrencyDetectionPhase.scanning,
        statusText: 'جاري المسح',
        debugText: '',
      ),
    );
    _pausedForBackground = false;
    _startPipelineWatchdog();
    _startPulsingVibration();
    unawaited(WakelockPlus.enable());
  }

  Future<void> stop({bool announce = true}) async {
    if (!state.isActive) {
      return;
    }
    _runId++;
    _pausedForBackground = false;
    _announcementInProgress = false;
    _stopPulsingVibration();
    _stopPipelineWatchdog();
    unawaited(WakelockPlus.disable());
    _cooldownTimer?.cancel();
    _cooldownTimer = null;
    _cooldownUntil = null;
    await _currencySubscription?.cancel();
    _currencySubscription = null;
    _voter.clear();
    _repositionGuidance.reset();
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
    _runId++;
    _pausedForBackground = false;
    _announcementInProgress = false;
    _stopPulsingVibration();
    _stopPipelineWatchdog();
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
    await _setTorch(false);
    _repository.dispose();
    return super.close();
  }
}
