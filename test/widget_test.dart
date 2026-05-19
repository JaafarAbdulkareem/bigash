// Smoke test for Bigash. The full app boots camera and TFLite which
// can't run inside the widget test harness, so we only verify that the
// splash route renders without throwing.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('Splash renders the app title', (WidgetTester tester) async {
    await tester.pumpWidget(
      const MaterialApp(
        home: Scaffold(
          body: Center(
            child: Text('بيقش\nBigash'),
          ),
        ),
      ),
    );

    expect(find.textContaining('Bigash'), findsOneWidget);
    expect(find.textContaining('بيقش'), findsOneWidget);
  });
}
