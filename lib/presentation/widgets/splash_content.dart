import 'package:flutter/material.dart';

/// Pure, accessibility-first presentation for the splash/permission screen.
class SplashContent extends StatelessWidget {
  const SplashContent({
    required this.status,
    required this.waitingForPermission,
    required this.fadeAnimation,
    this.onRetry,
    super.key,
  });

  final String status;
  final bool waitingForPermission;
  final Animation<double> fadeAnimation;
  final VoidCallback? onRetry;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      key: const Key('splash_action_surface'),
      onTap: waitingForPermission ? onRetry : null,
      behavior: HitTestBehavior.opaque,
      child: DecoratedBox(
        decoration: const BoxDecoration(
          gradient: RadialGradient(
            center: Alignment(0, -0.2),
            radius: 0.9,
            colors: <Color>[Color(0xFF1A1A2E), Colors.black],
          ),
        ),
        child: SafeArea(
          child: LayoutBuilder(
            builder: (BuildContext context, BoxConstraints constraints) {
              return SingleChildScrollView(
                child: ConstrainedBox(
                  constraints: BoxConstraints(minHeight: constraints.maxHeight),
                  child: Semantics(
                    container: true,
                    excludeSemantics: true,
                    label: waitingForPermission
                        ? 'إذن الكاميرا مطلوب. اضغط على الشاشة لإعادة المحاولة.'
                        : 'شاشة البداية لتطبيق بيقش. $status',
                    button: waitingForPermission,
                    child: FadeTransition(
                      opacity: fadeAnimation,
                      child: Padding(
                        padding: const EdgeInsets.all(24),
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: <Widget>[
                            Container(
                              padding: const EdgeInsets.all(18),
                              decoration: BoxDecoration(
                                shape: BoxShape.circle,
                                color: Colors.yellow.withValues(alpha: 0.12),
                                border: Border.all(
                                  color: Colors.yellow.withValues(alpha: 0.5),
                                  width: 2,
                                ),
                              ),
                              child: const Icon(
                                Icons.account_balance_wallet_outlined,
                                color: Colors.yellow,
                                size: 56,
                              ),
                            ),
                            const SizedBox(height: 28),
                            const Text(
                              'بقش',
                              textAlign: TextAlign.center,
                              style: TextStyle(
                                fontSize: 64,
                                fontWeight: FontWeight.bold,
                                color: Colors.yellow,
                                height: 1,
                                letterSpacing: 2,
                              ),
                            ),
                            const SizedBox(height: 4),
                            Text(
                              'Bigash',
                              textAlign: TextAlign.center,
                              style: TextStyle(
                                fontSize: 22,
                                fontWeight: FontWeight.w500,
                                color: Colors.yellow.withValues(alpha: 0.75),
                                letterSpacing: 4,
                              ),
                            ),
                            const SizedBox(height: 36),
                            Text(
                              status,
                              key: const Key('splash_status'),
                              textAlign: TextAlign.center,
                              style: const TextStyle(
                                fontSize: 18,
                                color: Colors.white,
                                height: 1.4,
                              ),
                            ),
                            if (waitingForPermission) ...<Widget>[
                              const SizedBox(height: 32),
                              const Icon(
                                Icons.touch_app,
                                color: Colors.yellow,
                                size: 64,
                              ),
                              const SizedBox(height: 8),
                              const Text(
                                'اضغط لإعادة المحاولة',
                                style: TextStyle(
                                  color: Colors.yellow,
                                  fontSize: 16,
                                ),
                              ),
                            ],
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
              );
            },
          ),
        ),
      ),
    );
  }
}
