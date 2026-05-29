import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show PlatformException;
import 'package:permission_handler/permission_handler.dart';
import 'package:vibration/vibration.dart';
import 'package:bigash/data/services/tts_service.dart';
import 'package:bigash/presentation/pages/scanner_page.dart';

class SplashPage extends StatefulWidget {
  final TTSService ttsService;

  const SplashPage({super.key, required this.ttsService});

  @override
  State<SplashPage> createState() => _SplashPageState();
}

class _SplashPageState extends State<SplashPage> {
  bool _waitingForPermission = false;
  /// True while a `Permission.camera.request()` is in flight. Without this
  /// guard a fast double-tap on "Retry" fires a second request before the
  /// first dialog dismisses, and the plugin throws:
  ///   PlatformException(PermissionHandler.PermissionManager,
  ///     A request for permissions is already running...)
  bool _requestingPermission = false;
  String _status = 'مرحبا بك في تطبيق بقش';

  @override
  void initState() {
    super.initState();
    _initAndRoute();
  }

  Future<void> _initAndRoute() async {
    await widget.ttsService.initialize();
    // The visible name is "بقش" but TTS engines mispronounce it (the leading
    // sukun is not articulable, so the engine garbles or speeds through it).
    // Spelling it "بيقش" in the spoken string yields the intended Yemeni
    // pronunciation reliably — the يـ gives the engine an /iː/ vowel anchor.
    await widget.ttsService.speak('مرحبا بك في تطبيق بيقش');

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

    if (mounted) {
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (_) => const ScannerPage()),
      );
    }
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
    PermissionStatus status = await Permission.camera.status;
    if (status.isPermanentlyDenied) {
      await widget.ttsService.speak(
        'تم رفض الإذن بشكل دائم. سأفتح صفحة الإعدادات الآن.',
      );
      await openAppSettings();
      return;
    }
    final bool granted = await _ensureCameraPermission();
    if (granted && mounted) {
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (_) => const ScannerPage()),
      );
    } else {
      await _alertNeedsPermission();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: GestureDetector(
        onTap: _waitingForPermission ? _onRetryTap : null,
        behavior: HitTestBehavior.opaque,
        child: Center(
          child: Semantics(
            label: _waitingForPermission
                ? 'إذن الكاميرا مطلوب. اضغط على الشاشة لإعادة المحاولة.'
                // Spoken by TalkBack/VoiceOver — spell as "بيقش" for the
                // same reason the TTS string does (see _initAndRoute).
                : 'شاشة البداية، جاري تحميل تطبيق بيقش',
            button: _waitingForPermission,
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: <Widget>[
                  const Text(
                    'بقش\nBigash',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 48,
                      fontWeight: FontWeight.bold,
                      color: Colors.yellow,
                    ),
                  ),
                  const SizedBox(height: 24),
                  Text(
                    _status,
                    textAlign: TextAlign.center,
                    style: const TextStyle(
                      fontSize: 18,
                      color: Colors.white,
                    ),
                  ),
                  if (_waitingForPermission) ...<Widget>[
                    const SizedBox(height: 32),
                    const Icon(Icons.touch_app,
                        color: Colors.yellow, size: 64),
                    const SizedBox(height: 8),
                    const Text(
                      'اضغط لإعادة المحاولة',
                      style: TextStyle(color: Colors.yellow, fontSize: 16),
                    ),
                  ],
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
