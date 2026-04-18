import 'package:bigash/domain/entities/currency_entity.dart';

class CurrencyModel extends CurrencyEntity {
  const CurrencyModel({
    required super.label,
    required super.displayName,
    required super.confidence,
    super.topK,
  });

  factory CurrencyModel.fromInference({
    required String label,
    required String displayName,
    required double confidence,
    List<MapEntry<String, double>> topK = const <MapEntry<String, double>>[],
  }) {
    return CurrencyModel(
      label: label,
      displayName: displayName,
      confidence: confidence,
      topK: topK,
    );
  }
}
