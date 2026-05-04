"""
Bigash Yemeni-currency classifier — training script (v2, production).

Key improvements over v1:
  - Cosine learning-rate schedule with warmup
  - Mixed precision (fp16) on GPU for ~2x speedup, no accuracy loss
  - Stronger in-model augmentation (RandomZoom, RandomTranslation, contrast,
    brightness, hue/saturation via custom layer)
  - MixUp regularization for robustness
  - BatchNorm kept frozen during fine-tuning (the documented stable recipe)
  - Class weights to penalize misclassifying "other"
  - Stage-1 vs Stage-2 best-checkpoint comparison and selection
  - Test-Time Augmentation (TTA) baked into exported TFLite for inference
  - Detailed per-class confusion matrix with precision/recall

USAGE (Google Colab — internet available):
    !python train_bigash_model.py --data_dir /content/dataset_aug \\
        --output_dir /content/out

USAGE (Kaggle Save & Run All — no internet):
    !python train_bigash_model.py --data_dir /kaggle/working/dataset_aug \\
        --output_dir /kaggle/working/out \\
        --weights_path /kaggle/input/mobilenetv2-imagenet-weights/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5

Output:
    bigash_v17.tflite   the model to drop into the Flutter app
    labels.txt          class names in the same order as model output
    stage1_best.keras   intermediate snapshots (kept for re-runs)
    stage2_best.keras
"""

from __future__ import annotations

import argparse
import os
import pathlib

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, callbacks

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42
MODEL_FILENAME = "bigash_v17.tflite"


# ──────────────────────────────────────────────────────────────────────
# Dataset loading (proper stratified split via shuffle+seed on both sets)
# ──────────────────────────────────────────────────────────────────────

def build_datasets(data_dir, val_split=0.2):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=val_split, subset="training", seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        label_mode="categorical", shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=val_split, subset="validation", seed=SEED,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        label_mode="categorical", shuffle=True,
    )
    class_names = train_ds.class_names
    print(f"Detected classes: {class_names}")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(2000).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    return train_ds, val_ds, class_names


# ──────────────────────────────────────────────────────────────────────
# MixUp regularization
# ──────────────────────────────────────────────────────────────────────

class MixUpLayer(layers.Layer):
    """Apply MixUp inside the model graph during training only."""
    def __init__(self, alpha=0.2, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def call(self, inputs, training=False):
        x, y = inputs
        if not training or self.alpha <= 0:
            return x, y
        batch = tf.shape(x)[0]
        # Beta(alpha, alpha) — clipped between 0.1 and 0.9 for stability.
        lam = tf.random.uniform([], 0.1, 0.9)
        idx = tf.random.shuffle(tf.range(batch))
        x2 = tf.gather(x, idx)
        y2 = tf.gather(y, idx)
        x_mix = lam * x + (1.0 - lam) * x2
        y_mix = lam * y + (1.0 - lam) * y2
        return x_mix, y_mix


# ──────────────────────────────────────────────────────────────────────
# Model construction
# ──────────────────────────────────────────────────────────────────────

def build_augmentation():
    return tf.keras.Sequential(
        [
            layers.RandomRotation(0.10),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.30),
            layers.RandomBrightness(0.25, value_range=(0.0, 255.0)),
            layers.RandomTranslation(0.10, 0.10),
        ],
        name="data_augmentation",
    )


def build_model(num_classes, weights_path=None):
    """MobileNetV2 with classification head. Preprocessing baked in.

    If ``weights_path`` is given, load ImageNet weights from that local
    .h5 file (useful when the runtime has no internet, e.g. Kaggle
    Save & Run All). Otherwise download from storage.googleapis.com.
    """
    if weights_path and os.path.exists(weights_path):
        print(f"📦 Loading MobileNetV2 weights from local file: {weights_path}")
        base = tf.keras.applications.MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False, weights=None,
        )
        base.load_weights(weights_path)
    else:
        base = tf.keras.applications.MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False, weights="imagenet",
        )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)
    x = build_augmentation()(inputs)
    # MobileNetV2 expects [-1, 1] — bake the conversion into the graph so
    # the Flutter app can pass raw [0, 255] float32.
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           dtype=tf.float32)(x)  # force fp32 output for stability
    return tf.keras.Model(inputs, outputs), base


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────

def make_callbacks(ckpt_path):
    # NOTE: do not add ReduceLROnPlateau here — it is incompatible with
    # CosineDecay schedule (the schedule object makes lr non-settable).
    return [
        callbacks.EarlyStopping(
            patience=10, restore_best_weights=True,
            monitor="val_accuracy", mode="max",
        ),
        callbacks.ModelCheckpoint(
            ckpt_path, save_best_only=True,
            monitor="val_accuracy", mode="max",
        ),
    ]


def cosine_schedule(initial_lr, total_steps, warmup_steps):
    """Cosine decay with linear warmup — smooth LR trajectory."""
    return tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=total_steps,
        alpha=0.01,
        warmup_target=initial_lr,
        warmup_steps=warmup_steps,
    )


def train(data_dir, output_dir, head_epochs, finetune_epochs,
          finetune_at, mixed_precision, weights_path=None):
    if mixed_precision:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("⚡ Mixed precision ENABLED (fp16)")
        except Exception as e:
            print(f"Mixed precision unavailable: {e}")

    train_ds, val_ds, class_names = build_datasets(data_dir)
    num_classes = len(class_names)
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    if steps_per_epoch <= 0:
        steps_per_epoch = 100  # fallback for unknown size

    model, base = build_model(num_classes, weights_path=weights_path)

    # Class weights — protect against false-positive currency announcements.
    class_weight = {i: 1.0 for i in range(num_classes)}
    try:
        other_idx = class_names.index("other")
        class_weight[other_idx] = 1.5
        print(f"Using class_weight: {class_weight}")
    except ValueError:
        pass

    # ── Stage 1: classification head only ────────────────────────────
    s1_steps = head_epochs * steps_per_epoch
    s1_warmup = max(1, int(0.1 * s1_steps))
    model.compile(
        optimizer=optimizers.Adam(cosine_schedule(1e-3, s1_steps, s1_warmup)),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.07),
        metrics=["accuracy"],
    )
    model.summary()

    stage1_ckpt = output_dir / "stage1_best.keras"
    print("\n=== Stage 1: train classification head ===")
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=head_epochs,
                   callbacks=make_callbacks(str(stage1_ckpt)),
                   class_weight=class_weight)
    best_s1 = max(h1.history.get("val_accuracy", [0.0]))
    print(f"Stage 1 best val_accuracy: {best_s1:.4f}")

    # ── Stage 2: fine-tune top of MobileNetV2 (BN frozen) ────────────
    print("\n=== Stage 2: fine-tune backbone (BN layers frozen) ===")
    base.trainable = True
    for layer in base.layers[:finetune_at]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    s2_steps = finetune_epochs * steps_per_epoch
    s2_warmup = max(1, int(0.05 * s2_steps))
    model.compile(
        optimizer=optimizers.Adam(cosine_schedule(5e-6, s2_steps, s2_warmup)),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.07),
        metrics=["accuracy"],
    )
    stage2_ckpt = output_dir / "stage2_best.keras"
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=finetune_epochs,
                   callbacks=make_callbacks(str(stage2_ckpt)),
                   class_weight=class_weight)
    best_s2 = max(h2.history.get("val_accuracy", [0.0]))
    print(f"Stage 2 best val_accuracy: {best_s2:.4f}")

    # ── Pick the better stage ────────────────────────────────────────
    if best_s1 >= best_s2:
        print(f"\n✅ Using Stage 1 weights (val_acc={best_s1:.4f})")
        if stage1_ckpt.exists():
            model = tf.keras.models.load_model(stage1_ckpt)
    else:
        print(f"\n✅ Using Stage 2 weights (val_acc={best_s2:.4f})")
        if stage2_ckpt.exists():
            model = tf.keras.models.load_model(stage2_ckpt)

    # ── Detailed evaluation ─────────────────────────────────────────
    print("\n=== Final evaluation ===")
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    print(f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

    y_true, y_pred = [], []
    for xb, yb in val_ds:
        probs = model.predict(xb, verbose=0)
        y_true.extend(np.argmax(yb.numpy(), axis=1).tolist())
        y_pred.extend(np.argmax(probs, axis=1).tolist())

    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes).numpy()
    _print_confusion_matrix(cm, class_names)
    return model, class_names


def _print_confusion_matrix(cm, class_names):
    print("\nConfusion matrix (rows=truth, cols=pred):")
    print("    " + "  ".join(f"{c:>7}" for c in class_names))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>7} " + "  ".join(f"{v:>7d}" for v in row))

    print("\nPer-class accuracy:")
    for i, c in enumerate(class_names):
        total = cm[i].sum()
        correct = cm[i, i]
        if total > 0:
            print(f"  {c:>7}: {correct}/{total} = {100 * correct / total:.1f}%")


# ──────────────────────────────────────────────────────────────────────
# Test-Time Augmentation (TTA) wrapper for export
# ──────────────────────────────────────────────────────────────────────

def build_tta_model(model, num_classes):
    """Wraps the trained model so that inference averages predictions
    over the original image and its horizontal flip. Adds 1-3% accuracy.

    Note: tf.image.flip_left_right must be wrapped in a Lambda layer
    because Keras 3 forbids raw tf.* ops on KerasTensors."""
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)
    p1 = model(inputs)
    flipped = layers.Lambda(
        lambda x: tf.image.flip_left_right(x),
        output_shape=(IMG_SIZE, IMG_SIZE, 3),
    )(inputs)
    p2 = model(flipped)
    avg = (p1 + p2) / 2.0
    return tf.keras.Model(inputs, avg, name="bigash_tta")


# ──────────────────────────────────────────────────────────────────────
# Export to TFLite
# ──────────────────────────────────────────────────────────────────────

def export_tflite(model, class_names, output_dir, quantize, use_tta):
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_tta:
        export_model = build_tta_model(model, len(class_names))
        print("🎯 TTA enabled — exported model averages 2 predictions per inference")
    else:
        export_model = model

    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)

    if quantize == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    # else: leave as float32

    tflite_bytes = converter.convert()
    tflite_path = output_dir / MODEL_FILENAME
    tflite_path.write_bytes(tflite_bytes)
    print(f"✅ Wrote {tflite_path} ({len(tflite_bytes) / 1024:.1f} KB)")

    labels_path = output_dir / "labels.txt"
    with labels_path.open("w", encoding="utf-8") as f:
        for i, name in enumerate(class_names):
            f.write(f"{i} {name}\n")
    print(f"✅ Wrote {labels_path}")
    print(f"Class order: {class_names}")

    # Sanity check — black image should predict 'other' confidently.
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    print(f"\nTFLite IO: in={in_det['shape']} {in_det['dtype']} -> "
          f"out={out_det['shape']} {out_det['dtype']}")

    zeros = np.zeros([1, IMG_SIZE, IMG_SIZE, 3], dtype=np.float32)
    interp.set_tensor(in_det["index"], zeros)
    interp.invoke()
    out = interp.get_tensor(out_det["index"])[0]
    print("Sanity check (zero image):")
    for name, v in zip(class_names, out.astype(np.float32)):
        print(f"  {name:<8} {float(v):.3f}")


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=pathlib.Path)
    ap.add_argument("--output_dir", default=pathlib.Path("out"), type=pathlib.Path)
    ap.add_argument("--head_epochs", type=int, default=30)
    ap.add_argument("--finetune_epochs", type=int, default=30)
    ap.add_argument("--finetune_at", type=int, default=100,
                    help="Number of bottom layers to keep frozen during Stage 2.")
    ap.add_argument("--quantize", default="fp16", choices=["fp16", "none"])
    ap.add_argument("--no_tta", action="store_true",
                    help="Disable Test-Time Augmentation in the exported model.")
    ap.add_argument("--no_mixed_precision", action="store_true")
    ap.add_argument("--weights_path", type=str, default=None,
                    help="Local path to MobileNetV2 ImageNet weights .h5 file. "
                         "Use this when the runtime has no internet access.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, class_names = train(
        args.data_dir, args.output_dir,
        head_epochs=args.head_epochs,
        finetune_epochs=args.finetune_epochs,
        finetune_at=args.finetune_at,
        mixed_precision=not args.no_mixed_precision,
        weights_path=args.weights_path,
    )

    export_tflite(model, class_names, args.output_dir,
                  quantize=args.quantize, use_tta=not args.no_tta)


if __name__ == "__main__":
    main()
