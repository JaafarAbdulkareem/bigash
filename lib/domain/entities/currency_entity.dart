import 'package:equatable/equatable.dart';
import 'package:bigash/domain/currency/frame_quality.dart';

class CurrencyEntity extends Equatable {
  final String label;
  final String displayName;
  final double confidence;
  final List<MapEntry<String, double>> topK;
  final CaptureQuality quality;
  final double brightness;
  final double sharpness;

  const CurrencyEntity({
    required this.label,
    required this.displayName,
    required this.confidence,
    this.topK = const <MapEntry<String, double>>[],
    this.quality = CaptureQuality.usable,
    this.brightness = 0,
    this.sharpness = 0,
  });

  @override
  List<Object?> get props => <Object?>[
    label,
    displayName,
    confidence,
    topK,
    quality,
    brightness,
    sharpness,
  ];
}
