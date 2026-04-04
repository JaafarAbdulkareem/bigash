import 'package:flutter_tts/flutter_tts.dart';

class TTSService {
  final FlutterTts _flutterTts;

  TTSService(this._flutterTts);

  Future<void> initialize() async {
    await _flutterTts.setLanguage('ar');
    await _flutterTts.setPitch(1.0);
    await _flutterTts.setSpeechRate(0.48);
    await _flutterTts.awaitSpeakCompletion(true);
  }

  Future<void> speak(String text) async {
    await _flutterTts.stop();
    await _flutterTts.speak(text);
  }

  Future<void> stop() async {
    await _flutterTts.stop();
  }
}
