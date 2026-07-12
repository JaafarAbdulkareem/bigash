import 'package:bigash/domain/entities/currency_entity.dart';
import 'package:bigash/domain/currency/frame_quality.dart';

class CurrencyModel extends CurrencyEntity {
  const CurrencyModel({
    required super.label,
    required super.displayName,
    required super.confidence,
    super.topK,
    super.quality,
    super.brightness,
    super.sharpness,
  });

  factory CurrencyModel.fromInference({
    required String label,
    required String displayName,
    required double confidence,
    List<MapEntry<String, double>> topK = const <MapEntry<String, double>>[],
    CaptureQuality quality = CaptureQuality.usable,
    double brightness = 0,
    double sharpness = 0,
  }) {
    return CurrencyModel(
      label: label,
      displayName: displayName,
      confidence: confidence,
      topK: topK,
      quality: quality,
      brightness: brightness,
      sharpness: sharpness,
    );
  }
}
