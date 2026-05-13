"""
Bigash Yemeni-currency classifier — training script v3 (PRODUCTION TARGET 95%+).

Major upgrades over v2:
  - Backbone choice: EfficientNetB0 (default, higher accuracy) or MobileNetV2.
  - Removed in-model augmentation (offline augment_dataset.py v4 handles it).
    The cleaner graph makes TFLite export trivial (no Lambda/RandomLayer bugs).
  - MixUp + CutMix applied as tf.data map operations (40% MixUp, 30% CutMix).
  - Longer training: 50 head + 40 fine-tune epochs (was 30 + 30).
  - Class weights auto-computed from training set sizes.
  - Smarter Stage 2: unfreezes more layers, slightly higher LR.
  - Saves stage1_best.keras and stage2_best.keras; pick whichever scored higher.

USAGE (Google Colab — internet available):
    !python train_bigash_model.py --data_dir /content/dataset_aug \\
        --output_dir /content/out

USAGE (Kaggle Save & Run All — no internet, MobileNetV2 weights provided):
    !python train_bigash_model.py --data_dir /kaggle/working/dataset_aug \\
        --output_dir /kaggle/working/out \\
        --backbone mobilenetv2 \\
        --weights_path /kaggle/input/datasets/king1981/mobilenetv2-imagenet-weights/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5

USAGE (EfficientNetB0 — needs internet OR EfficientNetB0 weights .h5):
    !python train_bigash_model.py --data_dir /content/dataset_aug \\
        --output_dir /content/out --backbone efficientnetb0

Output:
    bigash_v18.tflite   the model to drop into the Flutter app
    labels.txt          class names in the same order as model output
    stage1_best.keras   intermediate snapshots (loaded back in Colab for export)
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
BATCH_SIZE = 16  # was 32 — halved to stay within Kaggle's RAM budget
SEED = 42
MODEL_FILENAME = "bigash_v18.tflite"


# ──────────────────────────────────────────────────────────────────────
# Dataset loading
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
    return train_ds, val_ds, class_names


# ──────────────────────────────────────────────────────────────────────
# MixUp + CutMix (applied via tf.data map; works with any backbone)
# ──────────────────────────────────────────────────────────────────────

def mixup_batch(images, labels, alpha=0.4):
    """Standard MixUp: x_mix = lam*x + (1-lam)*x_perm, y_mix similarly."""
    batch = tf.shape(images)[0]
    # Sample lam from Beta(alpha, alpha) — use uniform 0.1..0.9 as a simple
    # bounded proxy (closer to bimodal Beta with low alpha but TF-safe).
    lam = tf.random.uniform([], 0.1, 0.9)
    idx = tf.random.shuffle(tf.range(batch))
    images_b = tf.gather(images, idx)
    labels_b = tf.gather(labels, idx)
    images_mix = lam * images + (1.0 - lam) * images_b
    labels_mix = lam * labels + (1.0 - lam) * labels_b
    return images_mix, labels_mix


def cutmix_batch(images, labels, alpha=1.0):
    """CutMix: copy a random rectangular patch from a shuffled sample
    into each image, mix labels proportionally to patch area."""
    batch = tf.shape(images)[0]
    h = tf.shape(images)[1]
    w = tf.shape(images)[2]

    lam = tf.random.uniform([], 0.2, 0.8)
    cut_ratio = tf.math.sqrt(1.0 - lam)
    cut_h = tf.cast(tf.cast(h, tf.float32) * cut_ratio, tf.int32)
    cut_w = tf.cast(tf.cast(w, tf.float32) * cut_ratio, tf.int32)

    cy = tf.random.uniform([], 0, h, dtype=tf.int32)
    cx = tf.random.uniform([], 0, w, dtype=tf.int32)
    y1 = tf.clip_by_value(cy - cut_h // 2, 0, h)
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, h)
    x1 = tf.clip_by_value(cx - cut_w // 2, 0, w)
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, w)

    idx = tf.random.shuffle(tf.range(batch))
    images_b = tf.gather(images, idx)
    labels_b = tf.gather(labels, idx)

    # Build a (1, h, w, 1) mask that is 1 outside the patch, 0 inside.
    mask = tf.ones((h, w), dtype=images.dtype)
    yy = tf.range(h)
    xx = tf.range(w)
    in_y = tf.logical_and(yy >= y1, yy < y2)
    in_x = tf.logical_and(xx >= x1, xx < x2)
    patch_mask = tf.cast(in_y[:, None] & in_x[None, :], images.dtype)
    mask = 1.0 - patch_mask
    mask = mask[None, :, :, None]  # broadcast to (B, H, W, C)

    images_mix = images * mask + images_b * (1.0 - mask)
    real_lam = 1.0 - tf.cast(
        (y2 - y1) * (x2 - x1), tf.float32) / tf.cast(h * w, tf.float32)
    real_lam = tf.cast(real_lam, labels.dtype)
    labels_mix = real_lam * labels + (1.0 - real_lam) * labels_b
    return images_mix, labels_mix


def augment_pipeline(images, labels):
    """Randomly apply MixUp (40%) or CutMix (30%); otherwise pass through.
    Probabilities chosen empirically — MixUp slightly preferred."""
    r = tf.random.uniform([])
    return tf.cond(
        r < 0.40,
        lambda: mixup_batch(images, labels),
        lambda: tf.cond(
            r < 0.70,
            lambda: cutmix_batch(images, labels),
            lambda: (images, labels),
        ),
    )


def prepare_datasets(train_ds, val_ds):
    # Conservative pipeline — Kaggle's process is OOM-killed if total RAM
    # spikes past ~30 GB. AUTOTUNE was opening too many parallel workers
    # that each held an in-flight batch in memory. Fixed small values keep
    # the working set bounded.
    train_ds = train_ds.shuffle(500)                                  # was 2000
    train_ds = train_ds.map(augment_pipeline, num_parallel_calls=2)   # was AUTOTUNE
    train_ds = train_ds.prefetch(2)                                   # was AUTOTUNE
    val_ds = val_ds.prefetch(2)                                       # was AUTOTUNE
    return train_ds, val_ds


# ──────────────────────────────────────────────────────────────────────
# Model construction
# ──────────────────────────────────────────────────────────────────────

def build_backbone(name, weights_path=None):
    """Returns (backbone_model, preprocess_layers).

    The preprocess_layers convert input in [0, 255] float32 to whatever
    range the backbone expects. They are baked into the exported graph
    so the Flutter app can keep passing raw [0, 255]."""
    name = name.lower()
    if name == "mobilenetv2":
        if weights_path and os.path.exists(weights_path):
            print(f"📦 Loading MobileNetV2 weights from {weights_path}")
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
        # MobileNetV2 expects [-1, 1].
        preprocess = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)
        return base, preprocess

    if name == "efficientnetb0":
        if weights_path and os.path.exists(weights_path):
            print(f"📦 Loading EfficientNetB0 weights from {weights_path}")
            base = tf.keras.applications.EfficientNetB0(
                input_shape=(IMG_SIZE, IMG_SIZE, 3),
                include_top=False, weights=None,
            )
            base.load_weights(weights_path)
        else:
            base = tf.keras.applications.EfficientNetB0(
                input_shape=(IMG_SIZE, IMG_SIZE, 3),
                include_top=False, weights="imagenet",
            )
        # EfficientNet has Normalization baked in — pass raw [0, 255].
        # We still insert an Identity-ish Rescaling for export uniformity.
        preprocess = layers.Rescaling(scale=1.0, offset=0.0)
        return base, preprocess

    raise ValueError(f"Unknown backbone: {name}")


def build_model(num_classes, backbone_name, weights_path=None):
    """Clean architecture — no in-model augmentation; offline pipeline
    (augment_dataset.py v4) already provides 12 variants per image."""
    base, preprocess = build_backbone(backbone_name, weights_path)
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)
    x = preprocess(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(192, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           dtype=tf.float32)(x)  # fp32 output for stability
    return tf.keras.Model(inputs, outputs), base


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────

def make_callbacks(ckpt_path):
    # ReduceLROnPlateau is incompatible with CosineDecay — don't add it.
    return [
        callbacks.EarlyStopping(
            patience=12, restore_best_weights=True,
            monitor="val_accuracy", mode="max",
        ),
        callbacks.ModelCheckpoint(
            ckpt_path, save_best_only=True,
            monitor="val_accuracy", mode="max",
        ),
    ]


def cosine_schedule(initial_lr, total_steps, warmup_steps):
    return tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=total_steps,
        alpha=0.01,
        warmup_target=initial_lr,
        warmup_steps=warmup_steps,
    )


def compute_class_weights(data_dir, class_names):
    """Inverse-frequency class weights, normalised to mean=1.0.
    Boosts rare classes; 'other' gets an extra +20% to discourage
    false-positive currency announcements."""
    counts = []
    for c in class_names:
        d = pathlib.Path(data_dir) / c
        counts.append(len([p for p in d.iterdir() if p.is_file()]))
    counts = np.array(counts, dtype=np.float64)
    weights = counts.mean() / counts
    weights = weights / weights.mean()  # normalise so mean is 1.0
    weight_dict = {i: float(w) for i, w in enumerate(weights)}
    if "other" in class_names:
        weight_dict[class_names.index("other")] *= 1.2
    return weight_dict


def train(data_dir, output_dir, head_epochs, finetune_epochs,
          finetune_at, mixed_precision, backbone_name, weights_path=None):
    if mixed_precision:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("⚡ Mixed precision ENABLED (fp16)")
        except Exception as e:
            print(f"Mixed precision unavailable: {e}")

    train_ds_raw, val_ds, class_names = build_datasets(data_dir)
    num_classes = len(class_names)
    steps_per_epoch = tf.data.experimental.cardinality(train_ds_raw).numpy()
    if steps_per_epoch <= 0:
        steps_per_epoch = 100

    # Prepare datasets AFTER computing cardinality (cardinality on the raw
    # ds is meaningful, but unknown after .map(augment_pipeline)).
    train_ds, val_ds = prepare_datasets(train_ds_raw, val_ds)

    model, base = build_model(num_classes, backbone_name, weights_path)
    print(f"\n🧠 Backbone: {backbone_name}")
    print(f"📊 Steps per epoch: {steps_per_epoch}")

    class_weight = compute_class_weights(data_dir, class_names)
    print(f"⚖️ Class weights: { {class_names[i]: round(v, 2) for i, v in class_weight.items()} }")

    # ── Stage 1: classification head only ────────────────────────────
    s1_steps = head_epochs * steps_per_epoch
    s1_warmup = max(1, int(0.08 * s1_steps))
    model.compile(
        optimizer=optimizers.Adam(cosine_schedule(1e-3, s1_steps, s1_warmup)),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.08),
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

    # ── Stage 2: fine-tune top of backbone (BN frozen) ───────────────
    print("\n=== Stage 2: fine-tune backbone (BN layers frozen) ===")
    base.trainable = True
    for layer in base.layers[:finetune_at]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    s2_steps = finetune_epochs * steps_per_epoch
    s2_warmup = max(1, int(0.05 * s2_steps))
    # Slightly higher fine-tune LR than v2 (5e-6 was very timid).
    model.compile(
        optimizer=optimizers.Adam(cosine_schedule(1e-5, s2_steps, s2_warmup)),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.08),
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
# TFLite export — uses the working concrete-function path
# ──────────────────────────────────────────────────────────────────────

def export_tflite(model, class_names, output_dir, quantize):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a serving function with a fixed [1, 224, 224, 3] signature.
    # This avoids the from_keras_model bug on TF 2.16+/Keras 3 and is
    # the same path that worked in our Colab rescue cell.
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, IMG_SIZE, IMG_SIZE, 3], dtype=tf.float32)
    ])
    def serving_fn(x):
        return model(x, training=False)

    concrete = serving_fn.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], model)

    if quantize == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    tflite_bytes = converter.convert()
    tflite_path = output_dir / MODEL_FILENAME
    tflite_path.write_bytes(tflite_bytes)
    print(f"✅ Wrote {tflite_path} ({len(tflite_bytes) / 1024 / 1024:.2f} MB)")

    labels_path = output_dir / "labels.txt"
    with labels_path.open("w", encoding="utf-8") as f:
        for i, name in enumerate(class_names):
            f.write(f"{i} {name}\n")
    print(f"✅ Wrote {labels_path}")
    print(f"Class order: {class_names}")

    # Sanity check
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
    print("Sanity check (zero image — expect 'other' highest):")
    for name, v in zip(class_names, out.astype(np.float32)):
        print(f"  {name:<8} {float(v):.3f}")


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=pathlib.Path)
    ap.add_argument("--output_dir", default=pathlib.Path("out"), type=pathlib.Path)
    ap.add_argument("--backbone", default="mobilenetv2",
                    choices=["mobilenetv2", "efficientnetb0"])
    ap.add_argument("--head_epochs", type=int, default=50)      # was 30
    ap.add_argument("--finetune_epochs", type=int, default=40)  # was 30
    ap.add_argument("--finetune_at", type=int, default=80,
                    help="Number of bottom layers kept frozen in Stage 2.")
    ap.add_argument("--quantize", default="fp16", choices=["fp16", "none"])
    ap.add_argument("--no_mixed_precision", action="store_true")
    ap.add_argument("--weights_path", type=str, default=None,
                    help="Local path to backbone ImageNet weights .h5 file.")
    ap.add_argument("--skip_export", action="store_true",
                    help="Skip TFLite export (do it in Colab afterwards).")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, class_names = train(
        args.data_dir, args.output_dir,
        head_epochs=args.head_epochs,
        finetune_epochs=args.finetune_epochs,
        finetune_at=args.finetune_at,
        mixed_precision=not args.no_mixed_precision,
        backbone_name=args.backbone,
        weights_path=args.weights_path,
    )

    if not args.skip_export:
        try:
            export_tflite(model, class_names, args.output_dir,
                          quantize=args.quantize)
        except Exception as e:
            print(f"\n⚠️ Export to TFLite failed: {e}")
            print("   Keras checkpoints saved in output dir — convert in Colab.")


if __name__ == "__main__":
    main()
