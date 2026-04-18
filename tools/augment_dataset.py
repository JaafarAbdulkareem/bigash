"""
Offline data augmentation for Bigash.

Reads a folder tree:
    dataset/
        other/  y50/  y100/  y200/  y250/  y500/  y1000/

Writes an expanded tree:
    dataset_aug/
        other/  (original files + N augmented variants per image)
        y50/    (same)
        ...

The augmentations simulate real phone-camera conditions for a blind user:
  - brightness  (dim room, bright sun, flash on/off)
  - contrast
  - slight rotation (hand tilt)
  - slight zoom in/out
  - perspective shift (not holding phone perfectly parallel)
  - JPEG compression artifacts
  - gaussian noise (low-light camera sensor)
  - gaussian blur (shaky hand / slight out-of-focus)

HOW TO USE (on Google Colab, after unzipping dataset):
    !python augment_dataset.py --input_dir /content/dataset \\
        --output_dir /content/dataset_aug --variants 5

    !python train_bigash_model.py --data_dir /content/dataset_aug \\
        --output_dir /content/out

With --variants 5 and ~2300 input images you end up with ~14,000 images.
Increase to 8 variants for ~21,000 images if you need more.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter
import numpy as np


VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _random_brightness(img: Image.Image, rng: random.Random) -> Image.Image:
    factor = rng.uniform(0.55, 1.45)
    return ImageEnhance.Brightness(img).enhance(factor)


def _random_contrast(img: Image.Image, rng: random.Random) -> Image.Image:
    factor = rng.uniform(0.7, 1.35)
    return ImageEnhance.Contrast(img).enhance(factor)


def _random_saturation(img: Image.Image, rng: random.Random) -> Image.Image:
    factor = rng.uniform(0.7, 1.3)
    return ImageEnhance.Color(img).enhance(factor)


def _random_rotate(img: Image.Image, rng: random.Random) -> Image.Image:
    angle = rng.uniform(-12.0, 12.0)
    return img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))


def _random_zoom(img: Image.Image, rng: random.Random) -> Image.Image:
    zoom = rng.uniform(0.85, 1.15)
    w, h = img.size
    new_w, new_h = int(w * zoom), int(h * zoom)
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    if zoom >= 1.0:
        # Crop center.
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        return resized.crop((left, top, left + w, top + h))
    # Pad with black.
    out = Image.new("RGB", (w, h), (0, 0, 0))
    out.paste(resized, ((w - new_w) // 2, (h - new_h) // 2))
    return out


def _random_perspective(img: Image.Image, rng: random.Random) -> Image.Image:
    w, h = img.size
    shift = int(min(w, h) * 0.08)

    def jitter() -> int:
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


def _random_noise(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.array(img).astype(np.int16)
    sigma = rng.uniform(4, 18)
    noise = np.random.normal(0, sigma, arr.shape).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _random_blur(img: Image.Image, rng: random.Random) -> Image.Image:
    radius = rng.uniform(0.3, 1.6)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _maybe_apply(img, fn, rng, p):
    if rng.random() < p:
        return fn(img, rng)
    return img


def augment_once(img: Image.Image, seed: int) -> Image.Image:
    rng = random.Random(seed)
    # Always apply the most impactful ones; apply others probabilistically.
    img = _random_brightness(img, rng)
    img = _random_contrast(img, rng)
    img = _maybe_apply(img, _random_saturation, rng, p=0.7)
    img = _maybe_apply(img, _random_rotate, rng, p=0.8)
    img = _maybe_apply(img, _random_zoom, rng, p=0.6)
    img = _maybe_apply(img, _random_perspective, rng, p=0.4)
    img = _maybe_apply(img, _random_noise, rng, p=0.5)
    img = _maybe_apply(img, _random_blur, rng, p=0.3)
    return img


def process_class(
    class_dir: Path,
    out_dir: Path,
    variants: int,
    jpeg_quality_range: tuple[int, int],
):
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = [p for p in class_dir.iterdir() if p.suffix.lower() in VALID_EXT]
    print(f"  [{class_dir.name}] {len(imgs)} originals -> "
          f"{len(imgs) * (variants + 1)} total")

    for idx, src_path in enumerate(imgs):
        try:
            img = Image.open(src_path).convert("RGB")
        except Exception as e:
            print(f"    skipping {src_path.name}: {e}")
            continue

        # 1) Copy original (converted to RGB / JPEG for uniformity).
        orig_out = out_dir / f"{idx:05d}_orig.jpg"
        img.save(orig_out, "JPEG", quality=92)

        # 2) Write N augmented variants.
        for v in range(variants):
            seed = hash((src_path.name, v)) & 0xFFFFFFFF
            aug = augment_once(img, seed=seed)
            q = random.Random(seed).randint(*jpeg_quality_range)
            aug_out = out_dir / f"{idx:05d}_aug{v}.jpg"
            aug.save(aug_out, "JPEG", quality=q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--variants", type=int, default=5,
                    help="Number of augmented copies per original image.")
    ap.add_argument("--jpeg_min", type=int, default=60)
    ap.add_argument("--jpeg_max", type=int, default=95)
    args = ap.parse_args()

    if args.output_dir.exists():
        print(f"Removing existing {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    class_dirs = sorted(p for p in args.input_dir.iterdir() if p.is_dir())
    print(f"Found {len(class_dirs)} classes.")
    for cd in class_dirs:
        process_class(
            cd,
            args.output_dir / cd.name,
            variants=args.variants,
            jpeg_quality_range=(args.jpeg_min, args.jpeg_max),
        )
    print("\nDone.")


if __name__ == "__main__":
    main()
