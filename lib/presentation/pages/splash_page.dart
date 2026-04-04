import 'dart:async';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:bigash/data/services/tts_service.dart';
import 'package:bigash/presentation/pages/scanner_page.dart';

class SplashPage extends StatefulWidget {
  final TTSService ttsService;

  const SplashPage({super.key, required this.ttsService});

  @override
  State<SplashPage> createState() => _SplashPageState();
}

class _SplashPageState extends State<SplashPage> {
  @override
  void initState() {
    super.initState();
    _initAndRoute();
  }

  Future<void> _initAndRoute() async {
    await widget.ttsService.initialize();
    await widget.ttsService.speak('مرحبا بك في تطبيق بقش');

    await [
      Permission.camera,
    ].request();

    await Future<void>.delayed(const Duration(seconds: 2));

    if (mounted) {
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (_) => const ScannerPage()),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Semantics(
          label: "شاشة البداية، جاري تحميل تطبيق بيقش",
          child: Text(
            'بيقش\nBigash',
            textAlign: TextAlign.center,
            style: const TextStyle(
              fontSize: 48,
              fontWeight: FontWeight.bold,
              color: Colors.yellow,
            ),
          ),
        ),
      ),
    );
  }
}
