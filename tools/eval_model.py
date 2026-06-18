"""
Bigash — model evaluation through the APP'S EXACT decision path.

Why this exists: val_accuracy during training does NOT tell you what the user
experiences. The app feeds raw [0,255], center-crops to a square, averages 4
rotations (TTA), groups y100_new/old -> y100 (and y200), then applies hard gates
(currencyThreshold 0.35, voteMargin 0.10, oodMargin 0.15, otherCeiling 0.12) to
decide currency / unknown / uncertain. This script replays a folder of REAL test
photos through that identical path so you can answer, before shipping to clients:

  1. Is the green Indian 5-rupee (and other foreign notes) REJECTED (never
     announced as a Yemeni denomination)?
  2. For each real Yemeni denomination, what fraction clears the app's gates
     (the real-world "read rate"), and how big are the confidence margins?

USAGE (Colab/desktop, needs tensorflow + pillow + numpy):
    python tools/eval_model.py --model out/bigash_v18.tflite --labels out/labels.txt \
        --test_dir /content/test_set

  --test_dir contains one subfolder per label, e.g.
      test_set/y50/...  test_set/y200_old/...  test_set/other/...
  Put the 5-rupee photos under test_set/other/ (or any name containing 'rupee'
  / 'foreign' — those are always expected to be REJECTED).

KEEP THE THRESHOLDS BELOW IN SYNC with lib/domain/currency/frame_decision.dart.
"""

from __future__ import annotations

import argparse
import re
import pathlib

import numpy as np
from PIL import Image
import tensorflow as tf

IMG_SIZE = 224

# ── Must mirror frame_decision.dart DecisionThresholds ──────────────────
CURRENCY_THRESHOLD = 0.50
VOTE_MARGIN = 0.10
OOD_MARGIN = 0.15
OTHER_CEILING = 0.12
OTHER_THRESHOLD = 0.28

_INDEX_PREFIX = re.compile(r"^\d+[\s_]+")
_IGNORED = {"other", "background"}


def normalize_label(raw: str) -> str:
    return _INDEX_PREFIX.sub("", raw.strip().lower()).strip()


def group_key(norm: str) -> str:
    if norm in ("y100_new", "y100_old"):
        return "y100"
    if norm in ("y200_new", "y200_old"):
        return "y200"
    return norm


def is_ignored(key: str) -> bool:
    return key in _IGNORED


def decode_app_style(path: pathlib.Path) -> np.ndarray:
    """RGB -> center-crop to square -> resize 224 -> raw [0,255] float32,
    matching TFLiteService (crop_to_aspect_ratio + raw normalization)."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side)).resize(
        (IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)  # [224,224,3] in [0,255]


def predict_tta(interp, in_idx, out_idx, base: np.ndarray) -> np.ndarray:
    """Average softmax over 4 rotations, exactly like the app's TTA."""
    acc = None
    for k in range(4):
        rot = np.rot90(base, k=k).copy()
        interp.set_tensor(in_idx, rot[None, ...].astype(np.float32))
        interp.invoke()
        probs = interp.get_tensor(out_idx)[0].astype(np.float64)
        acc = probs if acc is None else acc + probs
    return acc / 4.0


def grouped_distribution(probs: np.ndarray, labels: list[str]):
    sums: dict[str, float] = {}
    for p, lab in zip(probs, labels):
        g = group_key(normalize_label(lab))
        sums[g] = sums.get(g, 0.0) + float(p)
    return sorted(sums.items(), key=lambda kv: (-kv[1], kv[0]))


def classify(grouped):
    """Mirror frame_decision.dart classifyFrame: returns (verdict, key, conf)."""
    if not grouped:
        return "uncertain", "", 0.0
    other_prob = sum(v for k, v in grouped if is_ignored(k))
    currency = [(k, v) for k, v in grouped if not is_ignored(k)]
    if not currency:
        if other_prob >= OTHER_THRESHOLD:
            return "unknown", "other", other_prob
        return "uncertain", "", 0.0
    top_k, top_v = currency[0]
    second_v = currency[1][1] if len(currency) > 1 else 0.0
    confident = top_v >= CURRENCY_THRESHOLD
    beats_denom = (top_v - second_v) >= VOTE_MARGIN
    beats_other = (top_v - other_prob) >= OOD_MARGIN
    other_low = other_prob <= OTHER_CEILING
    if confident and beats_denom and beats_other and other_low:
        return "currency", top_k, top_v
    if other_prob >= OTHER_THRESHOLD and (other_prob - top_v) >= VOTE_MARGIN:
        return "unknown", "other", other_prob
    return "uncertain", "", 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=pathlib.Path)
    ap.add_argument("--labels", required=True, type=pathlib.Path)
    ap.add_argument("--test_dir", required=True, type=pathlib.Path,
                    help="One subfolder per label; foreign/rupee folders are "
                         "always expected to be REJECTED.")
    args = ap.parse_args()

    labels = [normalize_label(line) for line in
              args.labels.read_text(encoding="utf-8").splitlines() if line.strip()]
    interp = tf.lite.Interpreter(model_path=str(args.model))
    interp.allocate_tensors()
    in_idx = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]
    out_n = interp.get_output_details()[0]["shape"][-1]
    if out_n != len(labels):
        raise SystemExit(f"Model classes ({out_n}) != labels ({len(labels)}).")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    failures = 0
    print(f"Gates: thr={CURRENCY_THRESHOLD} margin={VOTE_MARGIN} "
          f"ood={OOD_MARGIN} otherCeil={OTHER_CEILING}\n")

    for class_dir in sorted(p for p in args.test_dir.iterdir() if p.is_dir()):
        imgs = [p for p in class_dir.iterdir() if p.suffix.lower() in exts]
        if not imgs:
            continue
        folder = class_dir.name.lower()
        expect_reject = ("rupee" in folder or "foreign" in folder
                         or group_key(normalize_label(folder)) == "other")
        expected = "other" if expect_reject else group_key(normalize_label(folder))

        counts = {"currency": 0, "unknown": 0, "uncertain": 0}
        correct = 0
        wrong_denoms = {}
        margins = []
        for path in imgs:
            try:
                base = decode_app_style(path)
            except Exception as e:
                print(f"  skip {path.name}: {e}")
                continue
            probs = predict_tta(interp, in_idx, out_idx, base)
            grouped = grouped_distribution(probs, labels)
            verdict, key, conf = classify(grouped)
            counts[verdict] += 1
            if verdict == "currency":
                margins.append(conf)
            if expect_reject:
                # ANY currency announcement on a foreign/other note is a failure.
                if verdict == "currency":
                    failures += 1
                    wrong_denoms[key] = wrong_denoms.get(key, 0) + 1
                else:
                    correct += 1
            else:
                if verdict == "currency" and key == expected:
                    correct += 1
                elif verdict == "currency":
                    wrong_denoms[key] = wrong_denoms.get(key, 0) + 1

        n = len(imgs)
        tag = "REJECT" if expect_reject else f"->{expected}"
        avg_margin = (sum(margins) / len(margins)) if margins else 0.0
        print(f"[{class_dir.name}] ({tag}) n={n}  "
              f"correct={correct}/{n} ({100*correct/max(1,n):.0f}%)  "
              f"currency={counts['currency']} unknown={counts['unknown']} "
              f"uncertain={counts['uncertain']}  avg_conf={avg_margin:.2f}")
        if wrong_denoms:
            label = "FALSE DENOMINATION" if expect_reject else "MIS-READ as"
            print(f"    ⚠️ {label}: "
                  + ", ".join(f"{k}x{v}" for k, v in sorted(wrong_denoms.items())))

    print()
    if failures:
        raise SystemExit(
            f"❌ {failures} foreign/other image(s) were announced as a Yemeni "
            f"denomination — DO NOT SHIP. Add more such notes to 'other' and retrain.")
    print("✅ No foreign/other image was announced as currency. Safe on the tested set.")


if __name__ == "__main__":
    main()
