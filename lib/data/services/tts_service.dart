import 'package:flutter/foundation.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:vibration/vibration.dart';

class TTSService {
  final FlutterTts _flutterTts;
  bool _initialized = false;
  bool _initFailed = false;
  bool _arabicVoiceAvailable = true;

  TTSService(this._flutterTts);

  bool get isReady => _initialized && !_initFailed;
  bool get initFailed => _initFailed;

  /// False when the device has no Arabic TTS voice installed — spoken
  /// announcements may then be silent or mispronounced.
  bool get arabicVoiceAvailable => _arabicVoiceAvailable;

  Future<void> initialize() async {
    if (_initialized) return;
    try {
      await _flutterTts.setLanguage('ar');
      // Natural pitch. Pitch-shifting down to fake a deeper voice badly
      // distorts Arabic pronunciation (it sounds like a non-native
      // speaker). A genuine male voice — if the device has one installed —
      // is selected below instead.
      await _flutterTts.setPitch(1.0);
      // A touch brisker — 0.48 felt slow. Still clear for a blind user.
      await _flutterTts.setSpeechRate(0.6);
      await _flutterTts.awaitSpeakCompletion(true);
      _initialized = true;

      // List the device's Arabic voices and select a male one if present.
      await _setupArabicVoice();

      // setLanguage() does not throw when no Arabic voice is installed —
      // it just yields silent or mispronounced speech. Verify explicitly,
      // since audio is the primary channel for a blind user.
      try {
        final dynamic arOk = await _flutterTts.isLanguageAvailable('ar');
        _arabicVoiceAvailable = arOk == true || arOk == 1;
        if (!_arabicVoiceAvailable) {
          debugPrint('[TTS] no Arabic voice installed — announcements may be '
              'silent. The device needs an Arabic TTS voice.');
        }
      } catch (e) {
        debugPrint('[TTS] isLanguageAvailable check failed: $e');
      }
    } catch (e) {
      // TTS initialization can fail on devices without a TTS engine
      // (some emulators, stripped ROMs). Mark as failed so callers can
      // fall back to vibration-only signaling.
      debugPrint('[TTS] initialize failed: $e');
      _initFailed = true;
    }
  }

  /// Selects the male Arabic voice. 'ar-xa-x-are' is the confirmed male
  /// Google Arabic voice on the target devices; the offline (local) variant
  /// is preferred so the app speaks without an internet connection.
  Future<void> _setupArabicVoice() async {
    try {
      final dynamic voices = await _flutterTts.getVoices;
      if (voices is! List) return;
      const List<String> preferred = <String>[
        'ar-xa-x-are-local',
        'ar-xa-x-are-network',
      ];
      for (final String wanted in preferred) {
        for (final dynamic v in voices) {
          if (v is! Map) continue;
          if ('${v['name'] ?? ''}' == wanted) {
            await _flutterTts.setVoice(<String, String>{
              'name': '${v['name']}',
              'locale': '${v['locale']}',
            });
            debugPrint('[TTS] male Arabic voice set: $wanted');
            return;
          }
        }
      }
      debugPrint('[TTS] male voice ar-xa-x-are not found — using default');
    } catch (e) {
      debugPrint('[TTS] voice setup skipped: $e');
    }
  }

  /// Speaks [text]. If TTS isn't available, emits a long vibration as a
  /// fallback so a blind user at least knows something was announced.
  Future<void> speak(String text) async {
    if (!_initialized) {
      await initialize();
    }
    if (_initFailed) {
      await _fallbackVibration();
      return;
    }
    try {
      await _flutterTts.stop();
      await _flutterTts.speak(text);
    } catch (e) {
      debugPrint('[TTS] speak failed: $e');
      // One retry — many devices fail the first call after backgrounding.
      try {
        await _flutterTts.stop();
        await _flutterTts.speak(text);
      } catch (e2) {
        debugPrint('[TTS] speak retry failed: $e2');
        await _fallbackVibration();
      }
    }
  }

  Future<void> _fallbackVibration() async {
    try {
      if (await Vibration.hasVibrator()) {
        // Distinct long-short-long pattern so the user knows TTS failed
        // (vs. the normal short pulses from the scanner).
        await Vibration.vibrate(
          pattern: <int>[0, 400, 150, 100, 150, 400],
        );
      }
    } catch (_) {}
  }

  Future<void> stop() async {
    if (!_initialized || _initFailed) return;
    try {
      await _flutterTts.stop();
    } catch (e) {
      debugPrint('[TTS] stop failed: $e');
    }
  }
}
