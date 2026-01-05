#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


def _write_jpg(path: Path, rgb: np.ndarray, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(rgb)
    image.save(path, format="JPEG", quality=quality, subsampling=0, optimize=True)


def _synthetic_outdoor(width: int, height: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    sky = np.stack([0.35 + 0.25 * (1 - y), 0.55 + 0.30 * (1 - y), 0.85 + 0.10 * (1 - y)], axis=-1)
    ground = np.stack([0.20 + 0.30 * y, 0.35 + 0.35 * y, 0.15 + 0.20 * y], axis=-1)
    blend = (y > 0.55).astype(np.float32)[:, :, None]
    base = sky * (1 - blend) + ground * blend
    base = np.repeat(base, width, axis=1)

    # Add mild texture/noise.
    noise = rng.normal(0.0, 0.03, size=(height, width, 3)).astype(np.float32)
    base = np.clip(base + noise, 0.0, 1.0)

    # Simple "sun".
    cx, cy, r = int(width * 0.78), int(height * 0.22), int(min(width, height) * 0.10)
    yy, xx = np.ogrid[:height, :width]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    base[mask] = np.clip(base[mask] + np.array([0.35, 0.25, 0.0], dtype=np.float32), 0.0, 1.0)

    return (base * 255.0 + 0.5).astype(np.uint8)


def _synthetic_portrait(width: int, height: int, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    img = Image.new("RGB", (width, height), (230, 235, 240))
    d = ImageDraw.Draw(img)

    # Hair/backdrop.
    d.rectangle([0, int(height * 0.55), width, height], fill=(205, 210, 215))
    d.ellipse([int(width * 0.25), int(height * 0.05), int(width * 0.75), int(height * 0.75)], fill=(45, 35, 25))

    # Face.
    face = [int(width * 0.33), int(height * 0.18), int(width * 0.67), int(height * 0.70)]
    d.ellipse(face, fill=(235, 200, 175))

    # Eyes.
    ex0, ey = int(width * 0.43), int(height * 0.40)
    ex1 = int(width * 0.57)
    ew, eh = int(width * 0.06), int(height * 0.03)
    for ex in [ex0, ex1]:
        d.ellipse([ex - ew, ey - eh, ex + ew, ey + eh], fill=(255, 255, 255))
        d.ellipse([ex - ew // 3, ey - eh // 2, ex + ew // 3, ey + eh // 2], fill=(20, 60, 90))

    # Mouth.
    mx0, my0 = int(width * 0.46), int(height * 0.60)
    mx1, my1 = int(width * 0.54), int(height * 0.63)
    d.arc([mx0, my0, mx1, my1], start=200, end=340, fill=(140, 60, 60), width=3)

    # Add subtle sensor noise and blur.
    arr = np.asarray(img).astype(np.float32) / 255.0
    n = rng.uniform(0.01, 0.03)
    arr = np.clip(arr + np.random.default_rng(seed + 1).normal(0.0, n, arr.shape).astype(np.float32), 0.0, 1.0)
    img = Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    return np.asarray(img)


def _synthetic_textured_object(width: int, height: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.random((height, width, 3), dtype=np.float32)

    # Add structured stripes.
    x = np.linspace(0.0, 8.0 * np.pi, width, dtype=np.float32)[None, :]
    stripes = (0.5 + 0.5 * np.sin(x)).astype(np.float32)
    base[..., 0] = np.clip(base[..., 0] * 0.4 + 0.6 * stripes, 0.0, 1.0)
    base[..., 1] = np.clip(base[..., 1] * 0.6 + 0.4 * (1.0 - stripes), 0.0, 1.0)

    # Add a shaded "object" disk.
    cx, cy, r = int(width * 0.50), int(height * 0.52), int(min(width, height) * 0.22)
    yy, xx = np.ogrid[:height, :width]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
    mask = dist <= r
    shade = np.clip(1.0 - dist / (r + 1e-6), 0.0, 1.0)
    for c in range(3):
        base[..., c][mask] = np.clip(0.15 + 0.85 * shade[mask], 0.0, 1.0)

    return (base * 255.0 + 0.5).astype(np.uint8)


def _low_light(rgb: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = Image.fromarray(rgb)
    img = ImageEnhance.Brightness(img).enhance(0.18)
    img = ImageEnhance.Contrast(img).enhance(1.15)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.clip(arr + rng.normal(0.0, 0.02, arr.shape).astype(np.float32), 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic fixture images.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for fixtures.")
    parser.add_argument("--seed", type=int, default=1337, help="Base RNG seed.")
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    # Keep the fixture directory deterministic (remove stray files from previous runs).
    exts = {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff", ".bmp"}
    for p in out_dir.iterdir():
        if p.is_file() and (p.suffix.lower() in exts or p.name == "MANIFEST.txt"):
            p.unlink()

    width, height = 768, 512

    # 1) Indoor: reuse ml-sharp teaser if present (more realistic).
    teaser = Path(__file__).resolve().parents[2] / "third_party/ml-sharp/data/teaser.jpg"
    if teaser.exists():
        img = Image.open(teaser).convert("RGB")
        img = img.resize((width, height), resample=Image.BICUBIC)
        img.save(out_dir / "indoor_teaser.jpg", format="JPEG", quality=95, subsampling=0, optimize=True)
    else:
        _write_jpg(out_dir / "indoor_teaser.jpg", _synthetic_outdoor(width, height, args.seed + 99))

    # 2) Outdoor.
    _write_jpg(out_dir / "outdoor_synthetic.jpg", _synthetic_outdoor(width, height, args.seed + 1))

    # 3) Portrait.
    _write_jpg(out_dir / "portrait_synthetic.jpg", _synthetic_portrait(width, height, args.seed + 2))

    # 4) Textured object.
    _write_jpg(
        out_dir / "textured_object_synthetic.jpg",
        _synthetic_textured_object(width, height, args.seed + 3),
    )

    # 5) Low-light.
    indoor = np.asarray(Image.open(out_dir / "indoor_teaser.jpg").convert("RGB"))
    _write_jpg(out_dir / "low_light.jpg", _low_light(indoor, args.seed + 4))

    # 6) Optional: local user photo fixture if present at repo root.
    extra_img = Path(__file__).resolve().parents[2] / "IMG_6221.jpg"
    if extra_img.exists():
        img = Image.open(extra_img)
        img = ImageOps.exif_transpose(img).convert("RGB")
        img = img.resize((width, height), resample=Image.BICUBIC)
        img.save(out_dir / "IMG_6221.jpg", format="JPEG", quality=95, subsampling=0, optimize=True)

    # 7) Optional: local HEIC fixture (keep original file to preserve EXIF focal length).
    extra_heic = Path(__file__).resolve().parents[2] / "IMG_6221.HEIC"
    if not extra_heic.exists():
        extra_heic = Path(__file__).resolve().parents[2] / "IMG_6221.heic"
    if extra_heic.exists():
        shutil.copy2(extra_heic, out_dir / "IMG_6221_heic.heic")

    # Write a small manifest for reproducibility.
    manifest_lines = [
        "fixtures:",
        f"  seed: {args.seed}",
        f"  generated_size: [{width}, {height}]",
        "  notes:",
        '    - "Optional local files may keep original dimensions (not resized)."',
        "  files:",
    ]
    exts = {".jpg", ".jpeg", ".png", ".heic", ".tif", ".tiff", ".bmp"}
    files = [p for p in out_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    for p in sorted(files, key=lambda x: x.name):
        manifest_lines.append(f"    - {p.name}")
    (out_dir / "MANIFEST.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
