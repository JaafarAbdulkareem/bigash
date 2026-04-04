import 'package:bigash/domain/entities/currency_entity.dart';

class CurrencyModel extends CurrencyEntity {
  const CurrencyModel({
    required super.label,
    required super.displayName,
    required super.confidence,
  });

  factory CurrencyModel.fromInference({
    required String label,
    required String displayName,
    required double confidence,
  }) {
    return CurrencyModel(
      label: label,
      displayName: displayName,
      confidence: confidence,
    );
  }
}
