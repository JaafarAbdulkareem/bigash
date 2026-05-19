"""
Bigash — offline data augmentation pipeline (v5, LEAK-FREE SPLIT).

Reads:  dataset/<class>/<images>
Writes: dataset_aug/train/<class>/  (originals + N augmented variants)
        dataset_aug/val/<class>/    (raw originals only — no augmentation)

v5 — critical correctness fix vs v4:
  - The train/val split now happens HERE, on the SOURCE images, BEFORE any
    augmentation. Previously augmentation ran first and train_bigash_model.py
    split the already-augmented folder — so augmented variants of the same
    photo leaked into both train and val, inflating val accuracy into
    fiction. Splitting the sources first is the only way the reported
    accuracy reflects real-world performance.
  - Validation images are kept RAW (no augmentation): val must measure how
    the model does on real photos, not how it survives our filters.
  - Use --val_split to control the held-out fraction (default 0.2).

v4 augmentation set (unchanged — applied to TRAINING images only):
  - 12 variants per image; rotation ±20° + 30% chance of 90°/180°/270°
  - motion/gaussian blur, color temperature, gamma, grayscale (8%)
  - JPEG quality 25-92, cutout 1-3 patches, perspective, lens distortion
  - shadows and specular-highlight clusters
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageOps
import numpy as np


VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_SIDE = 384  # pre-resize target — small enough for fast aug, big enough for 224 training


# ──────────────────────────────────────────────────────────────────────
# Photometric augmentations
# ──────────────────────────────────────────────────────────────────────

def _random_brightness(img, rng):
    return ImageEnhance.Brightness(img).enhance(rng.uniform(0.40, 1.60))


def _random_contrast(img, rng):
    return ImageEnhance.Contrast(img).enhance(rng.uniform(0.55, 1.50))


def _random_saturation(img, rng):
    return ImageEnhance.Color(img).enhance(rng.uniform(0.50, 1.50))


def _random_sharpness(img, rng):
    return ImageEnhance.Sharpness(img).enhance(rng.uniform(0.4, 2.0))


def _random_color_temperature(img, rng):
    """Warm / cool shift via R,B channel swing."""
    arr = np.asarray(img, dtype=np.int16)
    shift = int(rng.uniform(-40, 40))
    arr[..., 0] = np.clip(arr[..., 0] + shift, 0, 255)
    arr[..., 2] = np.clip(arr[..., 2] - shift, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def _random_gamma(img, rng):
    """Non-linear lighting (mid-tone shift)."""
    gamma = rng.uniform(0.55, 1.7)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.power(arr, gamma)
    return Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8))


def _random_grayscale(img, rng):
    """Convert to grayscale then back to RGB. Forces the model to rely on shape
    rather than colour, which helps low-light shots that look near-monochrome."""
    gs = ImageOps.grayscale(img)
    return Image.merge("RGB", (gs, gs, gs))


# ──────────────────────────────────────────────────────────────────────
# Geometric augmentations
# ──────────────────────────────────────────────────────────────────────

def _random_rotate(img, rng):
    return img.rotate(rng.uniform(-20.0, 20.0), resample=Image.BILINEAR,
                      expand=False, fillcolor=(0, 0, 0))


def _random_orthogonal_rotate(img, rng):
    """90°/180°/270° rotation — handles users holding the bill sideways/upside-down."""
    return img.rotate(rng.choice([90, 180, 270]), resample=Image.BILINEAR,
                      expand=False, fillcolor=(0, 0, 0))


def _random_zoom(img, rng):
    zoom = rng.uniform(0.75, 1.25)
    w, h = img.size
    new_w, new_h = max(1, int(w * zoom)), max(1, int(h * zoom))
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    if zoom >= 1.0:
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        return resized.crop((left, top, left + w, top + h))
    out = Image.new("RGB", (w, h), (0, 0, 0))
    out.paste(resized, ((w - new_w) // 2, (h - new_h) // 2))
    return out


def _random_shear(img, rng):
    factor = rng.uniform(-0.22, 0.22)
    w, h = img.size
    return img.transform(
        (w, h), Image.AFFINE,
        (1, factor, -factor * h / 2 if factor > 0 else 0,
         0, 1, 0),
        resample=Image.BILINEAR, fillcolor=(0, 0, 0),
    )


def _random_perspective(img, rng):
    w, h = img.size
    shift = int(min(w, h) * 0.15)  # was 0.10 — more aggressive tilt

    def jitter():
        return rng.randint(-shift, shift)

    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [
        (jitter(), jitter()),
        (w + jitter(), jitter()),
        (w + jitter(), h + jitter()),
        (jitter(), h + jitter()),
    ]
    coeffs = _perspective_coeffs(src, dst)
    return img.transform((w, h), Image.PERSPECTIVE, coeffs,
                         Image.BILINEAR, fillcolor=(0, 0, 0))


def _perspective_coeffs(src, dst):
    matrix = []
    for (x, y), (X, Y) in zip(dst, src):
        matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
    A = np.array(matrix, dtype=np.float64)
    B = np.array(src, dtype=np.float64).reshape(8)
    res, *_ = np.linalg.lstsq(A, B, rcond=None)
    return tuple(float(v) for v in res)


def _random_translate(img, rng):
    w, h = img.size
    tx = rng.randint(int(-w * 0.12), int(w * 0.12))
    ty = rng.randint(int(-h * 0.12), int(h * 0.12))
    return img.transform(
        (w, h), Image.AFFINE, (1, 0, tx, 0, 1, ty),
        resample=Image.BILINEAR, fillcolor=(0, 0, 0),
    )


def _random_lens_distortion(img, rng):
    """Barrel or pincushion distortion — simulates wide-angle phone lenses
    and slight camera lens imperfections. Vectorised remap via np.indices."""
    arr = np.asarray(img, dtype=np.uint8)
    h, w = arr.shape[:2]
    # k > 0 = barrel; k < 0 = pincushion
    k = rng.uniform(-0.15, 0.15)
    cx, cy = w / 2.0, h / 2.0
    norm = max(cx, cy)

    yy, xx = np.indices((h, w), dtype=np.float32)
    dx = (xx - cx) / norm
    dy = (yy - cy) / norm
    r2 = dx * dx + dy * dy
    factor = 1.0 + k * r2
    sx = (dx * factor) * norm + cx
    sy = (dy * factor) * norm + cy
    sx = np.clip(sx, 0, w - 1).astype(np.int32)
    sy = np.clip(sy, 0, h - 1).astype(np.int32)
    out = arr[sy, sx]
    return Image.fromarray(out)


# ──────────────────────────────────────────────────────────────────────
# Quality / camera-condition simulators
# ──────────────────────────────────────────────────────────────────────

def _random_motion_blur(img, rng):
    """Aggressive hand-shake blur — Gaussian + multi-directional shift."""
    radius = rng.uniform(0.5, 3.5)
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    angle = rng.uniform(0, 2 * math.pi)
    shift = rng.randint(2, 10)
    dx = int(round(shift * math.cos(angle)))
    dy = int(round(shift * math.sin(angle)))
    shifted = img.transform(
        img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy),
        resample=Image.BILINEAR, fillcolor=(0, 0, 0),
    )
    return Image.blend(img, shifted, 0.5)


def _random_gaussian_blur(img, rng):
    return img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 3.0)))


def _random_noise(img, rng):
    """Mix of Gaussian and Poisson-like noise (realistic camera noise)."""
    arr = np.array(img).astype(np.int16)
    sigma = rng.uniform(5, 28)
    noise = np.random.normal(0, sigma, arr.shape).astype(np.int16)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def _random_jpeg_quality(img, rng):
    """Re-encode at random JPEG quality for compression artifacts.
    Range 25-92 includes nasty WhatsApp-level compression."""
    import io
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=rng.randint(25, 92))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _random_lowres(img, rng):
    """Down-sample then up-sample — simulates low-res or far-distance."""
    w, h = img.size
    factor = rng.uniform(1.8, 5.0)
    small_w = max(1, int(w / factor))
    small_h = max(1, int(h / factor))
    return img.resize((small_w, small_h), Image.BILINEAR).resize(
        (w, h), Image.BILINEAR)


# ──────────────────────────────────────────────────────────────────────
# Occlusion / lighting simulators
# ──────────────────────────────────────────────────────────────────────

def _random_erasing(img, rng):
    """Cutout — 1 to 3 opaque rectangles simulating fingers/objects."""
    w, h = img.size
    area = w * h
    out = img.copy()
    draw = ImageDraw.Draw(out)
    num_cuts = rng.choice([1, 1, 2, 3])  # weighted: usually 1, sometimes 2-3
    for _ in range(num_cuts):
        target = rng.uniform(0.02, 0.30) * area / num_cuts
        aspect = rng.uniform(0.3, 3.3)
        rh = min(h - 1, int(math.sqrt(target / aspect)))
        rw = min(w - 1, int(math.sqrt(target * aspect)))
        if rh <= 0 or rw <= 0:
            continue
        x = rng.randint(0, w - rw)
        y = rng.randint(0, h - rh)
        color = (rng.randint(30, 230), rng.randint(30, 230), rng.randint(30, 230))
        draw.rectangle((x, y, x + rw, y + rh), fill=color)
    return out


def _random_shadow(img, rng):
    """1-2 smooth darkness gradients. Multiple shadows simulate complex lighting."""
    w, h = img.size
    arr = np.asarray(img, dtype=np.float32)
    num = rng.choice([1, 1, 2])
    for _ in range(num):
        direction = rng.choice(["left", "right", "top", "bottom"])
        intensity = rng.uniform(0.30, 0.80)
        if direction in ("left", "right"):
            ramp = np.linspace(intensity, 1.0, w, dtype=np.float32)
            if direction == "right":
                ramp = ramp[::-1]
            mask = np.broadcast_to(ramp, (h, w))
        else:
            ramp = np.linspace(intensity, 1.0, h, dtype=np.float32)
            if direction == "bottom":
                ramp = ramp[::-1]
            mask = np.broadcast_to(ramp[:, None], (h, w))
        arr = arr * mask[..., None]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def _random_highlight(img, rng):
    """1-3 bright spots simulating flash glare / specular reflections."""
    w, h = img.size
    arr = np.asarray(img, dtype=np.float32)
    num = rng.choice([1, 1, 2, 3])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for _ in range(num):
        cx = rng.randint(0, w)
        cy = rng.randint(0, h)
        radius = rng.uniform(min(w, h) * 0.10, min(w, h) * 0.35)
        intensity = rng.uniform(35, 95)
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        glow = np.exp(-dist_sq / (2.0 * radius * radius)) * intensity
        arr = arr + glow[..., None]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


# ──────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────

def _maybe(img, fn, rng, p):
    return fn(img, rng) if rng.random() < p else img


def _preresize(img):
    """Resize so that the long edge is MAX_SIDE px, preserving aspect."""
    w, h = img.size
    long_edge = max(w, h)
    if long_edge <= MAX_SIDE:
        return img
    scale = MAX_SIDE / long_edge
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                      Image.BILINEAR)


def augment_once(img, seed):
    rng = random.Random(seed)

    # Photometric
    img = _random_brightness(img, rng)
    img = _random_contrast(img, rng)
    img = _maybe(img, _random_saturation, rng, 0.7)
    img = _maybe(img, _random_color_temperature, rng, 0.5)
    img = _maybe(img, _random_gamma, rng, 0.45)
    img = _maybe(img, _random_sharpness, rng, 0.4)
    img = _maybe(img, _random_grayscale, rng, 0.08)  # rare, but useful

    # Geometric — orthogonal first (so subsequent rotations stack nicely)
    img = _maybe(img, _random_orthogonal_rotate, rng, 0.30)  # ← solves upside-down
    img = _maybe(img, _random_rotate, rng, 0.85)
    img = _maybe(img, _random_zoom, rng, 0.65)
    img = _maybe(img, _random_translate, rng, 0.5)
    img = _maybe(img, _random_shear, rng, 0.35)
    img = _maybe(img, _random_perspective, rng, 0.50)
    img = _maybe(img, _random_lens_distortion, rng, 0.25)  # new: lens curvature

    # Camera-quality degradations
    if rng.random() < 0.60:
        if rng.random() < 0.5:
            img = _random_motion_blur(img, rng)
        else:
            img = _random_gaussian_blur(img, rng)

    img = _maybe(img, _random_noise, rng, 0.55)
    img = _maybe(img, _random_shadow, rng, 0.35)
    img = _maybe(img, _random_highlight, rng, 0.30)
    img = _maybe(img, _random_erasing, rng, 0.35)
    img = _maybe(img, _random_lowres, rng, 0.30)
    img = _maybe(img, _random_jpeg_quality, rng, 0.55)

    return img


def process_class(class_dir, train_out, val_out, variants, val_split, seed):
    """Split this class's SOURCE images into train/val, THEN augment.

    Splitting before augmentation is the whole point of v5: augmented
    variants of one photo must never land in both train and val, or the
    reported val accuracy becomes meaningless."""
    train_out.mkdir(parents=True, exist_ok=True)
    val_out.mkdir(parents=True, exist_ok=True)

    imgs = sorted(p for p in class_dir.iterdir()
                  if p.suffix.lower() in VALID_EXT)
    if not imgs:
        print(f"  [{class_dir.name}] no images — skipped")
        return

    # Deterministic per-class shuffle, then hold out `val_split` of the
    # SOURCE images. Seeded so re-runs produce the exact same split.
    rng = random.Random(f"{seed}:{class_dir.name}")
    shuffled = imgs[:]
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_split))
    if len(shuffled) >= 2:
        n_val = min(max(n_val, 1), len(shuffled) - 1)  # never empty either side
    else:
        n_val = 0
    val_imgs = shuffled[:n_val]
    train_imgs = shuffled[n_val:]

    print(f"  [{class_dir.name}] {len(imgs)} originals -> "
          f"train {len(train_imgs)} (+{len(train_imgs) * variants} aug)  |  "
          f"val {len(val_imgs)} (raw)")
    if n_val == 0:
        print(f"    ⚠️ '{class_dir.name}' has too few images for a val split "
              f"— add more photos of this denomination.")

    # ── Training images: original + N augmented variants ─────────────
    for idx, src_path in enumerate(train_imgs):
        try:
            img = Image.open(src_path).convert("RGB")
            img = _preresize(img)
        except Exception as e:
            print(f"    skipping {src_path.name}: {e}")
            continue
        img.save(train_out / f"{idx:05d}_orig.jpg", "JPEG", quality=92)
        for v in range(variants):
            seed_v = hash((src_path.name, v)) & 0xFFFFFFFF
            try:
                aug = augment_once(img, seed=seed_v)
                aug.save(train_out / f"{idx:05d}_aug{v}.jpg", "JPEG",
                         quality=random.Random(seed_v).randint(70, 95))
            except Exception as e:
                print(f"    aug failed {src_path.name} v{v}: {e}")

    # ── Validation images: raw originals only, NO augmentation ───────
    for idx, src_path in enumerate(val_imgs):
        try:
            img = Image.open(src_path).convert("RGB")
            img = _preresize(img)
        except Exception as e:
            print(f"    skipping {src_path.name}: {e}")
            continue
        img.save(val_out / f"{idx:05d}_orig.jpg", "JPEG", quality=92)


def preview_augmentations(image_path, count=16, output_path="preview.png"):
    """Generate a grid showing N random augmentations of one image."""
    import matplotlib.pyplot as plt

    img = Image.open(image_path).convert("RGB")
    img = _preresize(img)
    cols = 4
    rows = (count // cols) + (1 if count % cols else 0)
    fig, axes = plt.subplots(rows + 1, cols, figsize=(cols * 3, (rows + 1) * 3))

    axes[0, 0].imshow(img)
    axes[0, 0].set_title("ORIGINAL", fontsize=10, color="green")
    for j in range(cols):
        axes[0, j].axis("off")

    for i in range(count):
        row = (i // cols) + 1
        col = i % cols
        aug = augment_once(img, seed=i + 1000)
        axes[row, col].imshow(aug)
        axes[row, col].set_title(f"variant {i + 1}", fontsize=9)
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=80, bbox_inches="tight")
    print(f"✅ Preview saved to {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=Path)
    ap.add_argument("--output_dir", type=Path)
    ap.add_argument("--variants", type=int, default=12)  # was 8
    ap.add_argument("--val_split", type=float, default=0.2,
                    help="Fraction of SOURCE images per class held out for "
                         "validation (split before augmentation — leak-free).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for the deterministic train/val split.")
    ap.add_argument("--preview", type=Path)
    args = ap.parse_args()

    if args.preview is not None:
        preview_augmentations(args.preview)
        return

    if args.input_dir is None or args.output_dir is None:
        ap.error("--input_dir and --output_dir are required.")

    if not 0.0 < args.val_split < 0.9:
        ap.error("--val_split must be between 0 and 0.9")

    if args.output_dir.exists():
        print(f"Removing existing {args.output_dir}")
        shutil.rmtree(args.output_dir)
    train_root = args.output_dir / "train"
    val_root = args.output_dir / "val"
    train_root.mkdir(parents=True)
    val_root.mkdir(parents=True)

    class_dirs = sorted(p for p in args.input_dir.iterdir() if p.is_dir())
    print(f"Found {len(class_dirs)} classes, {args.variants} variants each.")
    print(f"Val split: {args.val_split:.0%} of SOURCE images, held out "
          f"BEFORE augmentation (no leakage).")
    print(f"Pre-resize to {MAX_SIDE} px on long edge for speed.\n")
    for cd in class_dirs:
        process_class(cd, train_root / cd.name, val_root / cd.name,
                      variants=args.variants, val_split=args.val_split,
                      seed=args.seed)
    print(f"\nDone.\n  train -> {train_root}\n  val   -> {val_root}")


if __name__ == "__main__":
    main()
