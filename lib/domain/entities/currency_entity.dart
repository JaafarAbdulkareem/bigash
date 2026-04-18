import 'package:equatable/equatable.dart';

class CurrencyEntity extends Equatable {
  final String label;
  final String displayName;
  final double confidence;
  final List<MapEntry<String, double>> topK;

  const CurrencyEntity({
    required this.label,
    required this.displayName,
    required this.confidence,
    this.topK = const <MapEntry<String, double>>[],
  });

  @override
  List<Object?> get props => [label, displayName, confidence];
}
