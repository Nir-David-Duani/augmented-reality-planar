#!/usr/bin/env python3
"""
Convert HEIC-renamed-as-JPG files to real JPEG and optionally resize images.

Default behavior:
  - Reads from data/chessboard_jpg
  - Writes converted/resized JPEGs to data/chessboard_jpg_fixed
  - Leaves originals untouched

Usage examples:
  python tools/fix_chessboard_images.py
  python tools/fix_chessboard_images.py --size 4284 5712
  python tools/fix_chessboard_images.py --in_dir data/chessboard_jpg --out_dir data/chessboard_jpg_fixed
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
from typing import Iterable, Tuple

from PIL import Image

try:
    import pillow_heif  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pillow_heif = None


def is_heic_by_signature(data: bytes) -> bool:
    # HEIC files often contain ftypheic/ftyphevc/ftypmif1 in the header
    header = data[:64].lower()
    return b"ftypheic" in header or b"ftyphevc" in header or b"ftypmif1" in header


def iter_images(in_dir: Path) -> Iterable[Path]:
    for path in sorted(in_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic"}:
            yield path


def load_image(path: Path) -> Image.Image:
    data = path.read_bytes()
    if is_heic_by_signature(data):
        if pillow_heif is None:
            raise RuntimeError(
                "HEIC detected but pillow-heif is not installed. "
                "Install with: pip install pillow-heif"
            )
        heif = pillow_heif.open_heif(data)
        return Image.frombytes(heif.mode, heif.size, heif.data, "raw", heif.mode, heif.stride)
    return Image.open(io.BytesIO(data))


def maybe_resize(img: Image.Image, size: Tuple[int, int] | None) -> Image.Image:
    if size is None:
        return img
    if img.size == size:
        return img
    return img.resize(size, Image.LANCZOS)


def save_jpeg(img: Image.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = img.convert("RGB")
    rgb.save(out_path, format="JPEG", quality=95, optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fix chessboard images for calibration.")
    parser.add_argument("--in_dir", type=Path, default=Path("data/chessboard_jpg"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/chessboard_jpg_fixed"))
    parser.add_argument("--size", nargs=2, type=int, metavar=("W", "H"))
    args = parser.parse_args()

    if args.size is not None:
        target_size = (args.size[0], args.size[1])
    else:
        target_size = None

    in_dir = args.in_dir
    out_dir = args.out_dir
    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")

    converted = 0
    resized = 0
    skipped = 0

    for path in iter_images(in_dir):
        try:
            img = load_image(path)
        except Exception as exc:
            print(f"[skip] {path}: {exc}")
            skipped += 1
            continue

        resized_img = maybe_resize(img, target_size)
        if resized_img is not img:
            resized += 1

        out_path = out_dir / (path.stem + ".jpg")
        save_jpeg(resized_img, out_path)
        converted += 1

    print(f"done: converted={converted}, resized={resized}, skipped={skipped}")
    print(f"output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

