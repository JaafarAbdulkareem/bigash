"""
Bigash Yemeni-currency classifier — training script.

HOW TO USE (Google Colab, free GPU)
===================================

1) Go to https://colab.research.google.com  -> New Notebook
2) Runtime -> Change runtime type -> Hardware accelerator: GPU (T4)
3) Collect your dataset into a folder tree like this, then zip it:

        dataset/
            other/      <-- >=200 photos: white paper, hands, tables, random clutter,
                             backs of bills cropped weird, coins, receipts, etc.
                             THIS CLASS IS CRITICAL — without it the model will
                             confidently call any random image a bill.
            y50/        <-- 100+ photos of real 50 rial notes (front + back),
                             different lighting, angles, wrinkles, partial views
            y100/
            y200/
            y250/
            y500/
            y1000/

   Tips for good data:
   - Use the SAME phone camera you'll use in production if possible
   - Vary: lighting (bright, dim, flashlight on), background (wood, cloth, hand),
     angle (straight, tilted, rotated), zoom (full bill, half bill), bill condition
     (new, wrinkled, folded)
   - At least 100 photos per class, 200+ is better
   - Keep sizes reasonable (1000-2000px on the long side)

4) Upload the zip to Colab (left sidebar -> Files -> Upload).
   Or upload to Google Drive and mount it.

5) In a Colab cell:
        !unzip dataset.zip -d /content/
        !pip install -q tensorflow==2.15.0
        !python train_bigash_model.py --data_dir /content/dataset --output_dir /content/out

6) Download /content/out/bigash_v16.tflite and /content/out/labels.txt
   and replace the ones under assets/ in the Flutter project.

The output .tflite is float32 with [-1, 1] normalization — this matches
the Flutter app's default `NormalizationMode.signed`.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42


def build_datasets(data_dir: pathlib.Path, val_split: float = 0.2):
    # NOTE: shuffle=True on BOTH subsets with the same seed is required so that
    # image_dataset_from_directory shuffles the full file list before splitting.
    # With shuffle=False the validation subset just takes the tail of the
    # alphabetically-ordered file list, producing a broken non-stratified split.
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
    )
    class_names = train_ds.class_names
    print(f"Detected classes: {class_names}")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    return train_ds, val_ds, class_names


def build_model(num_classes: int) -> tf.keras.Model:
    data_aug = tf.keras.Sequential(
        [
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.12),
            layers.RandomContrast(0.25),
            layers.RandomBrightness(0.25),
            layers.RandomTranslation(0.08, 0.08),
        ],
        name="data_augmentation",
    )

    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)
    x = data_aug(inputs)
    # MobileNetV2 expects [-1, 1].
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), base


def train(
    data_dir: pathlib.Path,
    output_dir: pathlib.Path,
    head_epochs: int = 30,
    finetune_epochs: int = 25,
    finetune_at: int = 100,
    skip_finetune: bool = False,
):
    train_ds, val_ds, class_names = build_datasets(data_dir)
    num_classes = len(class_names)
    assert num_classes >= 2, "Need at least 2 classes."

    model, base = build_model(num_classes)
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    model.summary()

    stage1_ckpt = output_dir / "stage1_best.keras"
    stage2_ckpt = output_dir / "stage2_best.keras"

    def _callbacks(filepath: str):
        return [
            callbacks.EarlyStopping(
                patience=10, restore_best_weights=True,
                monitor="val_accuracy", mode="max",
            ),
            callbacks.ModelCheckpoint(
                filepath, save_best_only=True,
                monitor="val_accuracy", mode="max",
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", patience=4, factor=0.5, min_lr=1e-7,
            ),
        ]

    print("\n=== Stage 1: train classification head ===")
    hist1 = model.fit(
        train_ds, validation_data=val_ds, epochs=head_epochs,
        callbacks=_callbacks(str(stage1_ckpt)),
    )
    best_s1 = max(hist1.history.get("val_accuracy", [0.0]))
    print(f"Stage 1 best val_accuracy: {best_s1:.4f}")

    best_s2 = 0.0
    if not skip_finetune:
        print("\n=== Stage 2: fine-tune MobileNetV2 top layers ===")
        base.trainable = True
        # Unfreeze the tail of the backbone, but KEEP BatchNorm layers frozen
        # in inference mode. This is the documented fix for the usual
        # "fine-tuning makes val_accuracy drop" problem — BN stats should
        # not be updated on a small downstream dataset.
        for layer in base.layers[:finetune_at]:
            layer.trainable = False
        for layer in base.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        model.compile(
            optimizer=optimizers.Adam(5e-6),  # gentler LR
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
            metrics=["accuracy"],
        )
        hist2 = model.fit(
            train_ds, validation_data=val_ds, epochs=finetune_epochs,
            callbacks=_callbacks(str(stage2_ckpt)),
        )
        best_s2 = max(hist2.history.get("val_accuracy", [0.0]))
        print(f"Stage 2 best val_accuracy: {best_s2:.4f}")

    # Pick whichever stage produced the better validation accuracy.
    if best_s1 >= best_s2:
        print(f"\nUsing Stage 1 weights (val_acc={best_s1:.4f})")
        if stage1_ckpt.exists():
            model = tf.keras.models.load_model(stage1_ckpt)
    else:
        print(f"\nUsing Stage 2 weights (val_acc={best_s2:.4f})")
        if stage2_ckpt.exists():
            model = tf.keras.models.load_model(stage2_ckpt)

    # Evaluate and write confusion matrix.
    print("\n=== Final evaluation ===")
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    y_true, y_pred = [], []
    for xb, yb in val_ds:
        probs = model.predict(xb, verbose=0)
        y_true.extend(np.argmax(yb.numpy(), axis=1).tolist())
        y_pred.extend(np.argmax(probs, axis=1).tolist())
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes).numpy()
    print("Confusion matrix (rows=truth, cols=pred):")
    print("    " + "  ".join(f"{c:>6}" for c in class_names))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>6} " + "  ".join(f"{v:>6d}" for v in row))

    return model, class_names


def export_tflite(
    model: tf.keras.Model,
    class_names: list[str],
    output_dir: pathlib.Path,
    representative_dataset_dir: pathlib.Path | None = None,
    quantize: str = "fp16",  # "fp16" | "int8" | "none"
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build an inference-only graph that matches the Flutter preprocessing:
    #   - expects a uint8 [0,255] image in signed [-1,1] normalization happens in model.
    # To keep the Flutter side simple, we export a model that takes float32 in [-1, 1]
    # because the Flutter code already does `(pixel - 127.5) / 127.5`.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantize == "int8":
        assert representative_dataset_dir is not None, "int8 needs representative data"

        def rep_gen():
            for root, _, files in os.walk(representative_dataset_dir):
                for fn in files[:200]:
                    p = os.path.join(root, fn)
                    try:
                        img = tf.io.read_file(p)
                        img = tf.io.decode_image(img, channels=3, expand_animations=False)
                        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
                        img = tf.cast(img, tf.float32)
                        yield [tf.expand_dims(img, 0).numpy()]
                    except Exception:
                        continue

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    # else "none" — leave as float32.

    tflite_bytes = converter.convert()
    tflite_path = output_dir / "bigash_v16.tflite"
    tflite_path.write_bytes(tflite_bytes)
    print(f"Wrote {tflite_path} ({len(tflite_bytes)/1024:.1f} KB)")

    # Write labels.txt in the format the Flutter app already expects: "<idx> <name>"
    labels_path = output_dir / "labels.txt"
    with labels_path.open("w", encoding="utf-8") as f:
        for i, name in enumerate(class_names):
            f.write(f"{i} {name}\n")
    print(f"Wrote {labels_path}")
    print("Class order:", class_names)

    # Sanity check: run a zero image through and print predictions.
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    print(f"TFLite input : shape={in_det['shape']} dtype={in_det['dtype']}")
    print(f"TFLite output: shape={out_det['shape']} dtype={out_det['dtype']}")

    if in_det["dtype"] == np.float32:
        zeros = np.zeros([1, IMG_SIZE, IMG_SIZE, 3], dtype=np.float32)
    else:
        zeros = np.zeros([1, IMG_SIZE, IMG_SIZE, 3], dtype=np.uint8)
    interpreter.set_tensor(in_det["index"], zeros)
    interpreter.invoke()
    out = interpreter.get_tensor(out_det["index"])[0]
    print("Sanity check on zero image, per-class scores:")
    for name, v in zip(class_names, out.astype(np.float32)):
        print(f"  {name:<10} {float(v):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=pathlib.Path)
    ap.add_argument("--output_dir", default=pathlib.Path("out"), type=pathlib.Path)
    ap.add_argument("--quantize", default="fp16", choices=["fp16", "int8", "none"])
    ap.add_argument("--head_epochs", type=int, default=30)
    ap.add_argument("--finetune_epochs", type=int, default=25)
    ap.add_argument("--skip_finetune", action="store_true",
                    help="Only train the classifier head. Often enough.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, class_names = train(
        args.data_dir,
        args.output_dir,
        head_epochs=args.head_epochs,
        finetune_epochs=args.finetune_epochs,
        skip_finetune=args.skip_finetune,
    )
    export_tflite(
        model,
        class_names,
        args.output_dir,
        representative_dataset_dir=args.data_dir,
        quantize=args.quantize,
    )


if __name__ == "__main__":
    main()
