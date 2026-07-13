import 'package:bigash/presentation/cubits/detection_cubit.dart';
import 'package:bigash/presentation/widgets/scanner_surface.dart';
import 'package:bigash/presentation/widgets/splash_content.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('SplashContent exposes retry and survives large text', (
    WidgetTester tester,
  ) async {
    int retries = 0;
    await tester.pumpWidget(
      MaterialApp(
        home: MediaQuery(
          data: const MediaQueryData(textScaler: TextScaler.linear(3)),
          child: SplashContent(
            status: 'إذن الكاميرا مطلوب',
            waitingForPermission: true,
            fadeAnimation: const AlwaysStoppedAnimation<double>(1),
            onRetry: () => retries++,
          ),
        ),
      ),
    );

    expect(find.byKey(const Key('splash_status')), findsOneWidget);
    expect(find.text('اضغط لإعادة المحاولة'), findsOneWidget);
    expect(tester.takeException(), isNull);
    expect(
      tester.getCenter(find.text('بقش')).dx,
      closeTo(
        tester.getSize(find.byKey(const Key('splash_action_surface'))).width /
            2,
        0.5,
      ),
    );

    await tester.tap(find.byKey(const Key('splash_action_surface')));
    expect(retries, 1);
  });

  testWidgets('ScannerSurface renders diagnostic and success state', (
    WidgetTester tester,
  ) async {
    int taps = 0;
    const DetectionState state = DetectionState(
      isActive: true,
      isInitialized: true,
      statusText: 'خمسمائة ريال يمني',
      debugText: 'y500 92%',
      phase: CurrencyDetectionPhase.success,
    );

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: ScannerSurface(state: state, onTap: () => taps++),
        ),
      ),
    );

    expect(find.byKey(const Key('scanner_debug_overlay')), findsOneWidget);
    expect(find.text('خمسمائة ريال يمني'), findsOneWidget);
    expect(scannerPhaseColor(state), const Color(0xFF43A047));
    expect(scannerSemanticsLabel(state), contains('خمسمائة ريال يمني'));
    expect(
      tester
          .widget<Semantics>(find.byKey(const Key('scanner_semantics_surface')))
          .properties
          .liveRegion,
      isTrue,
    );

    await tester.tap(find.byKey(const Key('scanner_action_surface')));
    expect(taps, 1);
  });

  testWidgets('ScannerSurface idle state hides diagnostic overlay', (
    WidgetTester tester,
  ) async {
    final DetectionState state = DetectionState.initial();
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: ScannerSurface(state: state, onTap: () {}),
        ),
      ),
    );

    expect(find.byKey(const Key('scanner_debug_overlay')), findsNothing);
    expect(find.byKey(const Key('scanner_status_overlay')), findsOneWidget);
    expect(scannerPhaseColor(state), Colors.transparent);
    expect(
      tester
          .widget<Semantics>(find.byKey(const Key('scanner_semantics_surface')))
          .properties
          .liveRegion,
      isFalse,
    );
  });
}
