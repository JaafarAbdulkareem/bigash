# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this app is

**Bigash (بقش)** is an accessibility app that lets blind and visually-impaired
users identify **Yemeni riyal banknotes** by pointing the phone camera at a
bill. An on-device TFLite model classifies the denomination; the app announces
it with Arabic text-to-speech and signals state through vibration and the
camera torch. It is a humanitarian project — correctness of the spoken
denomination is safety-critical (a wrong announcement can cost a user money).

Every UI decision follows from "the user cannot see the screen": portrait is
locked, the whole screen is one gesture target, and feedback is audio + haptic.

## Commands

```bash
flutter pub get                      # install dependencies
flutter run                          # run on connected device/emulator
flutter analyze                      # lint (flutter_lints ruleset)
flutter test                         # run all tests
flutter test test/widget_test.dart   # run a single test file
flutter build apk --release          # release Android build
```

Model-training scripts live in `tools/` and run on Colab/Kaggle (GPU), not
locally — see the docstring at the top of each script for exact invocation:

```bash
python tools/scrape_other_images.py   # expand the "other" class from the web
python tools/augment_dataset.py       # leak-free train/val split + augment: dataset/ -> dataset_aug/{train,val}/
python tools/train_bigash_model.py    # train -> produces bigash_v18.tflite + labels.txt
```

## Architecture

Clean Architecture in three layers under `lib/`, wired by `get_it` in
[lib/core/di/injection_container.dart](lib/core/di/injection_container.dart).
State is managed with `flutter_bloc` cubits.

**Inference pipeline (the core data flow):**

1. `CameraService` ([lib/data/datasources/camera_service.dart](lib/data/datasources/camera_service.dart))
   picks the back camera, starts an image stream, exposes `frameStream`.
2. `CurrencyRepositoryImpl` ([lib/data/repositories/currency_repository_impl.dart](lib/data/repositories/currency_repository_impl.dart))
   subscribes `frameStream` and feeds each frame to `TFLiteService`.
3. `TFLiteService` ([lib/data/services/tflite_service.dart](lib/data/services/tflite_service.dart))
   converts the camera frame to RGB, rotates by `sensorOrientation`,
   center-crops, resizes to 224×224, runs inference (with 4-rotation TTA),
   softmaxes, and emits `TFLiteResult` on `resultStream`.
4. The repository maps each result to a `CurrencyEntity`, normalizing the raw
   label and attaching an Arabic TTS string via
   [lib/domain/currency/yemeni_currency_label_mapper.dart](lib/domain/currency/yemeni_currency_label_mapper.dart).
5. `DetectionCubit` ([lib/presentation/cubits/detection_cubit.dart](lib/presentation/cubits/detection_cubit.dart))
   applies majority voting over a rolling window, controls torch/vibration,
   enforces a speech cooldown, and triggers `TTSService` for the announcement.

`DetectionCubit` is where most behavior lives — voting, confidence thresholds,
phase machine (`idle`/`scanning`/`lowLight`/`success`), torch logic, and the
app-lifecycle pause/resume that keeps the camera handle alive in the background.

UI is just two pages: `SplashPage` (TTS welcome + camera-permission flow) and
`ScannerPage` (full-screen gesture surface: tap = start/stop, long-press =
screenshot to gallery). There is intentionally **no** runtime gesture to change
the normalization mode — cycling it away from `raw` would break the baked
`Rescaling` contract and mis-classify every bill.

## Model contract — easy to break, hard to notice

The model and the Dart preprocessing must agree exactly, or the app
"works" but mis-classifies. When touching the model or `TFLiteService`:

- **Normalization must match training.** `bigash_v18.tflite` has a `Rescaling`
  layer baked into its graph (see `build_backbone` in `train_bigash_model.py`),
  so the app feeds **raw [0,255] float32** — `TFLiteService.normalization`
  defaults to `NormalizationMode.raw`. Retraining with different preprocessing
  means changing this default.
- **Image geometry must match.** `train_bigash_model.py` loads images with
  `crop_to_aspect_ratio=True` (center-crop to square, then resize to 224×224);
  `TFLiteService` center-crops every camera frame the same way. Both sides
  must change together, or every prediction is geometrically wrong.
- **Never split an augmented dataset.** `augment_dataset.py` splits the source
  images into `train/`/`val/` *before* augmenting and leaves `val/` un-augmented;
  `train_bigash_model.py` consumes those folders directly. Splitting after
  augmentation leaks variants of the same photo across the boundary and makes
  the reported validation accuracy fictitious.
- **`labels.txt` order must match the model's output class index order** — the
  order printed as "Class order" by `train_bigash_model.py`. Line N of
  `assets/labels.txt` is class index N. Current classes (9, after the y100/y200
  new/old split): `other, y1000, y100_new, y100_old, y200_new, y200_old, y250,
  y50, y500`. `TFLiteService.initialize()` now asserts the model's output class
  count equals the labels line count and refuses to run (speaking an Arabic
  error) on a mismatch — so a labels/model desync fails loud instead of
  silently mis-mapping denominations.
- The model filename is referenced in **two** places — `pubspec.yaml` `assets:`
  and `TFLiteService.modelAssetPath`. Renaming the model means updating both.
- `labels.txt` entries are tolerated as `"<index> <label>"` or just `"<label>"`;
  `normalizeYemeniClassifierLabel` strips the leading index. Any label not in
  `arabicTtsForYemeniBill` falls through to being spoken verbatim.

Camera-frame decoding in `TFLiteService` is platform-specific: YUV420 on
Android, BGRA8888 on iOS.
