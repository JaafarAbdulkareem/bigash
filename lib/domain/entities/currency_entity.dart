import 'package:equatable/equatable.dart';

class CurrencyEntity extends Equatable {
  final String label;
  final String displayName;
  final double confidence;

  const CurrencyEntity({
    required this.label,
    required this.displayName,
    required this.confidence,
  });

  @override
  List<Object?> get props => [label, displayName, confidence];
}
