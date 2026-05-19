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
      await _flutterTts.setSpeechRate(0.48);
      await _flutterTts.awaitSpeakCompletion(true);
      _initialized = true;

      // Best-effort: pick an explicitly male Arabic voice when one exists.
      await _trySelectMaleArabicVoice();

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

  /// Picks an Arabic voice that is explicitly male when the platform
  /// exposes one. If the device only ships a female Arabic voice, no app
  /// setting can synthesise a male one — the lowered pitch is then the best
  /// available, and the user must install a male Arabic voice on the device.
  Future<void> _trySelectMaleArabicVoice() async {
    try {
      final dynamic voices = await _flutterTts.getVoices;
      if (voices is! List) return;
      for (final dynamic v in voices) {
        if (v is! Map) continue;
        final String locale = '${v['locale'] ?? ''}'.toLowerCase();
        if (!locale.startsWith('ar')) continue;
        // Engines expose gender in the name, a gender field, or the
        // identifier — check whatever is present.
        final String blob =
            ('${v['name'] ?? ''} ${v['gender'] ?? ''} ${v['identifier'] ?? ''}')
                .toLowerCase();
        debugPrint('[TTS] arabic voice available: $blob');
        if (blob.contains('male') && !blob.contains('female')) {
          await _flutterTts.setVoice(<String, String>{
            'name': '${v['name']}',
            'locale': '${v['locale']}',
          });
          debugPrint('[TTS] male Arabic voice selected: ${v['name']}');
          return;
        }
      }
      debugPrint('[TTS] no male Arabic voice on this device — pitch only');
    } catch (e) {
      debugPrint('[TTS] male-voice selection skipped: $e');
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
