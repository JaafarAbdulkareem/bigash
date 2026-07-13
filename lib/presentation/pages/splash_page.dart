import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show PlatformException;
import 'package:permission_handler/permission_handler.dart';
import 'package:vibration/vibration.dart';
import 'package:bigash/data/services/tts_service.dart';
import 'package:bigash/presentation/pages/scanner_page.dart';
import 'package:bigash/presentation/widgets/splash_content.dart';

class SplashPage extends StatefulWidget {
  final TTSService ttsService;

  const SplashPage({super.key, required this.ttsService});

  @override
  State<SplashPage> createState() => _SplashPageState();
}

class _SplashPageState extends State<SplashPage>
    with SingleTickerProviderStateMixin, WidgetsBindingObserver {
  bool _waitingForPermission = false;

  /// Drives the entrance fade-in for the splash content. Partially-sighted
  /// users see "something appeared", blind users are unaffected. Duration
  /// is short on purpose — we don't want to delay the camera prompt.
  late final AnimationController _fadeController = AnimationController(
    duration: const Duration(milliseconds: 650),
    vsync: this,
  )..forward();
  late final Animation<double> _fadeIn = CurvedAnimation(
    parent: _fadeController,
    curve: Curves.easeOut,
  );

  /// True while a `Permission.camera.request()` is in flight. Without this
  /// guard a fast double-tap on "Retry" fires a second request before the
  /// first dialog dismisses, and the plugin throws:
  ///   PlatformException(PermissionHandler.PermissionManager,
  ///     A request for permissions is already running...)
  bool _requestingPermission = false;
  bool _navigating = false;
  String _status = 'مرحبا بك في تطبيق بقش';

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initAndRoute();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // When the user returns from the system Settings page (where they may have
    // just granted camera permission), re-check and proceed automatically — a
    // blind user shouldn't have to guess they must tap the screen again.
    if (state == AppLifecycleState.resumed && _waitingForPermission) {
      _recheckPermissionOnResume();
    }
  }

  Future<void> _recheckPermissionOnResume() async {
    final PermissionStatus status = await Permission.camera.status;
    if (status.isGranted) {
      await _openScanner();
    }
  }

  Future<void> _openScanner() async {
    if (!mounted || _navigating) return;
    _navigating = true;
    _waitingForPermission = false;
    await Navigator.of(context).pushReplacement(
      MaterialPageRoute<void>(builder: (_) => const ScannerPage()),
    );
  }

  Future<void> _initAndRoute() async {
    await widget.ttsService.initialize();
    // Audio is the ONLY safe channel for a blind user. If the engine failed to
    // init or the device has no Arabic voice, denomination announcements would
    // be silent/garbled — warn loudly (vibration is the reliable cue) instead
    // of silently promising readiness.
    if (widget.ttsService.initFailed ||
        !widget.ttsService.arabicVoiceAvailable) {
      await _warnTtsDegraded();
    } else {
      // The visible name is "بقش" but TTS engines mispronounce it (the leading
      // sukun is not articulable, so the engine garbles or speeds through it).
      // Spelling it "بيقش" in the spoken string yields the intended Yemeni
      // pronunciation reliably — the يـ gives the engine an /iː/ vowel anchor.
      await widget.ttsService.speak('مرحبا بك في تطبيق بيقش');
    }

    final bool granted = await _ensureCameraPermission();
    if (!granted) {
      // Permission denied — stay on splash and keep prompting. Blind users
      // need clear audio feedback that they must grant camera access.
      if (!mounted) return;
      setState(() {
        _waitingForPermission = true;
        _status = 'يرجى السماح للتطبيق باستخدام الكاميرا للتعرف على العملة';
      });
      await _alertNeedsPermission();
      return;
    }

    await Future<void>.delayed(const Duration(milliseconds: 800));

    await _openScanner();
  }

  /// Requests camera permission and returns whether it was granted.
  /// Distinguishes between `denied` (we can ask again) and `permanentlyDenied`
  /// (user must open Settings).
  Future<bool> _ensureCameraPermission() async {
    PermissionStatus status = await Permission.camera.status;
    if (status.isGranted) {
      return true;
    }
    // Re-entrancy guard: if another request is already in flight (e.g. a
    // rapid second tap on the Retry surface), do not fire a parallel one
    // — the plugin throws if it sees two requests overlap.
    if (_requestingPermission) {
      return false;
    }
    _requestingPermission = true;
    try {
      await widget.ttsService.speak(
        'ستظهر الآن نافذة إذن الكاميرا. استخدم قارئ الشاشة، ثم اختر أثناء استخدام التطبيق، '
        'أو هذه المرة فقط. كلا الخيارين يسمحان بقراءة العملة.',
      );
      if (!mounted) return false;
      status = await Permission.camera.request();
    } on PlatformException {
      // Shouldn't happen now that we guard above, but if the OS races us
      // anyway, treat it as "not granted yet" and let the user retry.
      return false;
    } finally {
      _requestingPermission = false;
    }
    return status.isGranted;
  }

  /// Warns the user (vibration + best-effort speech + on-screen text) that the
  /// device cannot speak Arabic, so denomination announcements may be silent.
  /// The vibration is the reliable signal; the speech may itself be silent.
  Future<void> _warnTtsDegraded() async {
    if (await Vibration.hasVibrator()) {
      // Long-short-long: a distinct "something is wrong" pattern, different
      // from the scanner's short pulses.
      unawaited(Vibration.vibrate(pattern: <int>[0, 500, 200, 150, 200, 500]));
    }
    if (mounted) {
      setState(() {
        _status =
            'تنبيه: لا يتوفر صوت عربي للنطق على هذا الجهاز. '
            'ثبّت صوتا عربيا من إعدادات الجهاز لقراءة فئات العملة بوضوح.';
      });
    }
    await widget.ttsService.speak(
      'تنبيه: لا يوجد صوت عربي للنطق على هذا الجهاز. '
      'يرجى تثبيت صوت عربي من إعدادات الجهاز لقراءة فئات العملة بوضوح.',
    );
  }

  Future<void> _alertNeedsPermission() async {
    // Loud audio + strong vibration so a blind user notices even with the
    // phone in their pocket / on a table.
    if (await Vibration.hasVibrator()) {
      unawaited(Vibration.vibrate(pattern: <int>[0, 200, 100, 200, 100, 200]));
    }
    await widget.ttsService.speak(
      'لم يتم السماح للتطبيق باستخدام الكاميرا. '
      'اضغط على الشاشة لإعادة المحاولة، '
      'أو افتح إعدادات الجهاز لمنح الإذن يدويا.',
    );
  }

  Future<void> _onRetryTap() async {
    if (_requestingPermission || _navigating) return;
    PermissionStatus status = await Permission.camera.status;
    if (status.isPermanentlyDenied) {
      await widget.ttsService.speak(
        'تم رفض الإذن بشكل دائم. سأفتح صفحة الإعدادات الآن.',
      );
      await openAppSettings();
      return;
    }
    final bool granted = await _ensureCameraPermission();
    if (granted) {
      await _openScanner();
    } else {
      await _alertNeedsPermission();
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _fadeController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SplashContent(
        status: _status,
        waitingForPermission: _waitingForPermission,
        fadeAnimation: _fadeIn,
        onRetry: _onRetryTap,
      ),
    );
  }
}
