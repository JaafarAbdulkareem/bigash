"""
Bigash — automatic scraper to expand the "other" class.

Downloads images from Bing for a curated list of foreign currencies
and non-currency objects, filters them by size/aspect-ratio, and copies
the survivors into `<dataset>/other/`. Includes near-duplicate removal.

USAGE on Google Colab:
    !pip install -q bing-image-downloader imagehash
    !python scrape_other_images.py --dataset_dir /content/dataset \\
        --tmp_dir /content/scraped

By default the script targets ~600+ new images across foreign currencies
plus generic "non-currency" objects (paper, hands, receipts, fabric).
Use --extra_queries to add more targets (one per line in a text file).
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

from PIL import Image


# ──────────────────────────────────────────────────────────────────────
# Curated query list — focused on the gaps Bigash v16 had
# ──────────────────────────────────────────────────────────────────────

DEFAULT_QUERIES = [
    # ── Indian rupees — biggest source of false positives in v17 ──
    # Modern series (post-demonetization, common in Yemen)
    ("new 500 rupee note india", 60),
    ("500 rupee note red fort back", 40),
    ("Mahatma Gandhi 500 rupee note front", 40),
    ("indian 500 rupee back side", 30),
    ("new 200 rupee indian note", 50),
    ("new 200 rupee back red", 30),
    ("new 100 rupee indian note", 50),
    ("new 100 rupee back rani ki vav", 30),
    ("2000 rupee note india", 35),
    ("new 50 rupee indian note", 35),
    ("indian 20 rupee note new", 30),
    ("indian 10 rupee note new", 25),
    # OLD Mahatma Gandhi series (still circulating)
    ("old 500 rupee note india gandhi", 40),
    ("old 100 rupee note india gandhi", 30),
    ("old 50 rupee note india gandhi", 25),
    ("old 20 rupee note india gandhi", 25),
    # Various angles
    ("indian rupee note held in hand", 25),
    ("indian rupee note on table", 25),
    ("crumpled indian rupee note", 20),

    # ── Saudi riyal — both series (very common in Yemen) ──
    ("saudi riyal 500 banknote", 35),
    ("saudi riyal 500 back", 25),
    ("saudi riyal 100 note new", 35),
    ("saudi riyal 100 back side", 25),
    ("saudi riyal 50 banknote", 30),
    ("saudi riyal 20 banknote", 25),
    ("saudi riyal 10 banknote", 30),
    ("saudi riyal 5 note", 25),
    ("saudi riyal 1 note", 20),
    ("old saudi riyal note king fahd", 25),

    # ── Other Gulf / Arab currencies ──
    ("UAE dirham 100 note", 30),
    ("UAE dirham 50 note", 25),
    ("UAE dirham 20 note", 20),
    ("UAE dirham 10 note", 20),
    ("Egyptian pound 200 note", 25),
    ("Egyptian pound 100 note", 30),
    ("Egyptian pound 50 note", 25),
    ("Egyptian pound 20 note", 20),
    ("Jordanian dinar 20 note", 25),
    ("Jordanian dinar 50 note", 20),
    ("Omani rial 20 banknote", 25),
    ("Omani rial 50 banknote", 20),
    ("Kuwaiti dinar 20 note", 25),
    ("Kuwaiti dinar 10 note", 20),
    ("Qatar riyal note", 25),
    ("Bahraini dinar note", 20),
    ("Iraqi dinar 25000", 20),
    ("Iraqi dinar 10000", 20),
    ("Syrian pound note", 20),
    ("Lebanese pound 100000", 20),

    # ── USD / EUR / GBP ──
    ("us dollar 100 bill front", 30),
    ("us dollar 100 bill back", 25),
    ("us dollar 20 bill front", 25),
    ("us dollar 20 bill back", 20),
    ("us dollar 50 bill", 25),
    ("us dollar 10 bill", 20),
    ("us dollar 5 bill", 20),
    ("us dollar 1 bill", 20),
    ("euro 100 banknote", 30),
    ("euro 50 banknote", 25),
    ("euro 20 banknote", 20),
    ("euro 10 banknote", 20),
    ("british pound sterling 20 note", 20),

    # ── Asian currencies (common in Yemen via migrant workers) ──
    ("Pakistan rupee 1000 note", 25),
    ("Pakistan rupee 500 note", 25),
    ("Pakistan rupee 100 note", 20),
    ("Bangladesh 1000 taka note", 25),
    ("Bangladesh 500 taka note", 25),
    ("Bangladesh 100 taka note", 25),
    ("Sri Lankan rupee 5000 note", 20),
    ("Sri Lankan rupee 1000 note", 20),
    ("Philippine peso 1000 note", 20),
    ("Indonesian rupiah 100000 note", 20),
    ("Indonesian rupiah 50000 note", 20),

    # ── East African currencies ──
    ("Ethiopian birr 200 note", 25),
    ("Ethiopian birr 100 note", 25),
    ("Ethiopian birr 50 note", 20),
    ("Somali shilling note", 20),
    ("Kenyan shilling 1000 note", 20),

    # ── Generic non-currency (papers, hands, objects) ──
    ("blank white paper sheet", 30),
    ("white printer paper close up", 25),
    ("hand holding white paper", 30),
    ("crumpled receipt paper", 25),
    ("wooden table top texture", 25),
    ("colored cloth fabric texture", 20),
    ("blue jeans denim texture", 15),
    ("hand on wooden surface", 20),
    ("hand on tile floor", 15),
    ("playing cards face up", 25),
    ("business card photo", 20),
    ("torn paper edge", 20),
    ("colored sticky note", 15),
    ("magazine page texture", 20),
    ("newspaper page", 20),
    ("cardboard texture", 20),
    ("ceramic floor tile texture", 15),
    ("carpet rug texture", 15),
    ("smartphone screen black", 15),
    ("blurry photo paper", 15),

    # ── Other "looks like currency but isn't" ──
    ("paper coupon voucher", 20),
    ("bus ticket paper", 15),
    ("lottery ticket paper", 15),
    ("plane boarding pass paper", 15),
    ("identity card plastic", 15),
    ("credit card photo", 15),
    ("supermarket receipt paper", 20),
]


# ──────────────────────────────────────────────────────────────────────
# Download
# ──────────────────────────────────────────────────────────────────────

def download_all(queries, tmp_dir):
    from bing_image_downloader import downloader

    os.makedirs(tmp_dir, exist_ok=True)
    success, failures = 0, 0
    for query, limit in queries:
        print(f"🔍 {query}  (target {limit})")
        try:
            downloader.download(
                query, limit=limit, output_dir=str(tmp_dir),
                adult_filter_off=True, force_replace=False,
                timeout=60, verbose=False,
            )
            success += 1
        except Exception as e:
            print(f"   ⚠️ {e}")
            failures += 1
    print(f"\n📥 Queries: {success} succeeded, {failures} failed")


# ──────────────────────────────────────────────────────────────────────
# Validate, copy into dataset/other, and de-duplicate
# ──────────────────────────────────────────────────────────────────────

def copy_and_filter(tmp_dir, target_dir, min_aspect=1.15, max_aspect=4.0,
                    min_side=100, max_pixels=3000 * 3000):
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    moved, skipped = 0, 0

    for query_dir in os.listdir(tmp_dir):
        src_dir = tmp_dir / query_dir
        if not src_dir.is_dir():
            continue
        safe_query = query_dir.replace(" ", "_")[:25]
        for i, fname in enumerate(os.listdir(src_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in (".jpg", ".jpeg", ".png", ".webp"):
                skipped += 1
                continue
            src = src_dir / fname
            try:
                with Image.open(src) as img:
                    w, h = img.size
                    aspect = max(w, h) / min(w, h)
                    if (w < min_side or h < min_side or
                            aspect < min_aspect or aspect > max_aspect or
                            w * h > max_pixels):
                        skipped += 1
                        continue
            except Exception:
                skipped += 1
                continue

            dst = target_dir / f"scraped_{timestamp}_{safe_query}_{i:03d}{ext}"
            try:
                shutil.copy2(src, dst)
                moved += 1
            except Exception:
                skipped += 1
    print(f"\n✅ Moved {moved} valid images, skipped {skipped}")
    return moved


def remove_visual_duplicates(target_dir, threshold=6):
    """Use perceptual hashing to remove near-duplicate images."""
    import imagehash

    files = [
        f for f in os.listdir(target_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]
    hashes = {}
    for f in files:
        try:
            hashes[f] = imagehash.phash(Image.open(target_dir / f))
        except Exception:
            pass

    to_delete = set()
    items = list(hashes.items())
    for i, (f1, h1) in enumerate(items):
        if f1 in to_delete:
            continue
        for f2, h2 in items[i + 1:]:
            if f2 in to_delete:
                continue
            if abs(h1 - h2) < threshold:
                to_delete.add(f2)

    for f in to_delete:
        os.remove(target_dir / f)
    print(f"🗑️ Removed {len(to_delete)} visual duplicates")


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def load_extra_queries(path):
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Format: "query|limit"  or  just "query" (default limit 25)
            if "|" in line:
                q, n = line.split("|", 1)
                queries.append((q.strip(), int(n.strip())))
            else:
                queries.append((line, 25))
    return queries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True, type=Path,
                    help="Path to dataset root (containing 'other/' subfolder).")
    ap.add_argument("--tmp_dir", default=Path("/content/scraped"), type=Path)
    ap.add_argument("--extra_queries", type=Path,
                    help="Optional file with extra search terms (one per line).")
    ap.add_argument("--skip_download", action="store_true",
                    help="Skip the download step (use cached scraped/ folder).")
    args = ap.parse_args()

    queries = list(DEFAULT_QUERIES)
    if args.extra_queries and args.extra_queries.exists():
        extra = load_extra_queries(args.extra_queries)
        queries.extend(extra)
        print(f"📋 Loaded {len(extra)} extra queries from {args.extra_queries}")
    print(f"📋 Total queries: {len(queries)}")

    if not args.skip_download:
        download_all(queries, args.tmp_dir)

    target_dir = args.dataset_dir / "other"
    copy_and_filter(args.tmp_dir, target_dir)
    remove_visual_duplicates(target_dir)

    final_count = len(os.listdir(target_dir))
    print(f"\n📊 Final 'other' count: {final_count}")


if __name__ == "__main__":
    main()
