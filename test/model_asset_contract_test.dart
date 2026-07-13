import 'package:bigash/domain/currency/model_asset_contract.dart';
import 'package:crypto/crypto.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('bundled model has the reviewed SHA-256 digest', () async {
    final ByteData data = await rootBundle.load('assets/bigash_v18.tflite');
    final Uint8List bytes = data.buffer.asUint8List(
      data.offsetInBytes,
      data.lengthInBytes,
    );
    expect(sha256.convert(bytes).toString(), expectedBigashModelSha256);
  });

  test('bundled labels have the reviewed class order', () async {
    final String content = await rootBundle.loadString('assets/labels.txt');
    expect(labelsMatchExpectedBigashOrder(content.split('\n')), isTrue);
  });

  test('same labels in a different order are rejected', () {
    final List<String> reordered = List<String>.from(expectedBigashLabels);
    final String first = reordered.removeAt(0);
    reordered.add(first);
    expect(labelsMatchExpectedBigashOrder(reordered), isFalse);
  });
}
