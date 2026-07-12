import 'dart:async';

import 'package:bigash/data/services/tts_service.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_tts/flutter_tts.dart';

class FakeFlutterTts extends FlutterTts {
  bool failNextLanguageSetup = false;
  Completer<void>? stopGate;
  final List<String> spoken = <String>[];

  @override
  Future<dynamic> setLanguage(String language) async {
    if (failNextLanguageSetup) {
      failNextLanguageSetup = false;
      throw StateError('transient TTS failure');
    }
    return 1;
  }

  @override
  Future<dynamic> setPitch(double pitch) async => 1;

  @override
  Future<dynamic> setSpeechRate(double rate) async => 1;

  @override
  Future<dynamic> awaitSpeakCompletion(bool awaitCompletion) async => 1;

  @override
  Future<dynamic> get getVoices async => <dynamic>[];

  @override
  Future<dynamic> isLanguageAvailable(String language) async => 1;

  @override
  Future<dynamic> stop() async {
    await stopGate?.future;
    return 1;
  }

  @override
  Future<dynamic> speak(String text, {bool focus = false}) async {
    spoken.add(text);
    return 1;
  }
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('initialization can recover after a transient engine failure', () async {
    final FakeFlutterTts engine = FakeFlutterTts()
      ..failNextLanguageSetup = true;
    final TTSService service = TTSService(engine);

    await service.initialize();
    expect(service.initFailed, isTrue);

    await service.initialize();
    expect(service.isReady, isTrue);
    expect(service.initFailed, isFalse);
  });

  test('a newer utterance supersedes one still waiting to start', () async {
    final FakeFlutterTts engine = FakeFlutterTts();
    final TTSService service = TTSService(engine);
    await service.initialize();

    final Completer<void> gate = Completer<void>();
    engine.stopGate = gate;
    final Future<void> oldSpeech = service.speak('قديم');
    await Future<void>.delayed(Duration.zero);
    final Future<void> newSpeech = service.speak('جديد');
    gate.complete();

    await Future.wait(<Future<void>>[oldSpeech, newSpeech]);
    expect(engine.spoken, <String>['جديد']);
  });
}
