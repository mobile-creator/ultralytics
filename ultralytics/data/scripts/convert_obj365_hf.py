#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Convert Objects365 HuggingFace parquet dataset to Ultralytics YOLO format.

Usage:
    # Inspect schema first (recommended before conversion):
    python convert_obj365_hf.py --input /data/datasets/Objects365_hf/data --inspect

    # Convert (default: bbox format is xyxy, categories are 0-indexed):
    python convert_obj365_hf.py --input /data/datasets/Objects365_hf/data --output /data/datasets/Objects365

    # If your dataset uses COCO xywh bbox format (x_min, y_min, width, height):
    python convert_obj365_hf.py --input /data/datasets/Objects365_hf/data --output /data/datasets/Objects365 --bbox-format xywh

    # If your dataset uses 1-indexed category IDs (shift to 0-indexed):
    python convert_obj365_hf.py --input /data/datasets/Objects365_hf/data --output /data/datasets/Objects365 --cat-offset 1
"""

from __future__ import annotations

import argparse
import io
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def inspect_schema(parquet_path: Path) -> None:
    """Print schema and a sample row from a parquet file."""
    table = pq.read_table(parquet_path, columns=None)
    print(f"\nSchema for {parquet_path.name}:")
    print(table.schema)
    print(f"\nRow count: {table.num_rows}")
    row = table.slice(0, 1).to_pydict()
    for col, val in row.items():
        v = val[0]
        if isinstance(v, (bytes, bytearray)):
            print(f"  {col}: <bytes len={len(v)}>")
        elif isinstance(v, dict):
            summary = {k: (f"<bytes len={len(vv)}>" if isinstance(vv, (bytes, bytearray)) else vv) for k, vv in v.items()}
            print(f"  {col}: {summary}")
        else:
            print(f"  {col}: {v!r}")


def _bbox_to_yolo(bbox: list[float], img_w: int, img_h: int, is_xywh: bool) -> tuple[float, float, float, float]:
    """Convert bbox to normalized YOLO cx cy w h format."""
    x1, y1, x2, y2 = bbox
    if is_xywh:
        cx, cy, bw, bh = x1 + x2 / 2, y1 + y2 / 2, x2, y2
    else:
        cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    return cx / img_w, cy / img_h, bw / img_w, bh / img_h


def _process_shard(args: tuple) -> tuple[int, int]:
    """Process one parquet shard — returns (images_written, labels_written)."""
    parquet_path, images_dir, labels_dir, bbox_format, cat_offset = args
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    is_xywh = bbox_format == "xywh"

    table = pq.read_table(parquet_path)
    data = table.to_pydict()
    n = table.num_rows

    images_written = labels_written = 0
    for i in range(n):
        # --- Image ---
        img_field = data["image"][i]
        img_bytes = img_field["bytes"] if isinstance(img_field, dict) else img_field
        img_path_str = (img_field.get("path") or "") if isinstance(img_field, dict) else ""

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_w, img_h = img.size

        stem = Path(img_path_str).stem if img_path_str else f"{parquet_path.stem}_{i:06d}"
        img_out = images_dir / f"{stem}.jpg"
        img.save(img_out, "JPEG", quality=95)
        images_written += 1

        # --- Labels ---
        objs = data["objects"][i]
        if not objs:
            continue
        bboxes = objs.get("bbox", [])
        cats = objs.get("category", objs.get("label", []))
        crowds = objs.get("is_crowd", [False] * len(bboxes))

        lines = []
        for bbox, cat, crowd in zip(bboxes, cats, crowds):
            if crowd:
                continue
            cx, cy, bw, bh = _bbox_to_yolo(bbox, img_w, img_h, is_xywh)
            if bw <= 0 or bh <= 0:
                continue
            cls = int(cat) - cat_offset
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if lines:
            (labels_dir / f"{stem}.txt").write_text("\n".join(lines))
            labels_written += 1

    return images_written, labels_written


def convert(input_dir: Path, output_dir: Path, bbox_format: str, cat_offset: int, workers: int) -> None:
    """Convert all parquet shards in input_dir to YOLO format under output_dir."""
    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {input_dir}")

    # Group by split
    splits: dict[str, list[Path]] = {}
    for p in parquet_files:
        split = "val" if p.name.startswith("validation") else "train"
        splits.setdefault(split, []).append(p)

    for split, shards in splits.items():
        images_dir = output_dir / "images" / split
        labels_dir = output_dir / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        shard_args = [(p, images_dir, labels_dir, bbox_format, cat_offset) for p in shards]
        total_imgs = total_lbls = 0

        print(f"\nProcessing {split} split ({len(shards)} shards) ...")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_process_shard, a): a[0].name for a in shard_args}
            for fut in as_completed(futures):
                name = futures[fut]
                imgs, lbls = fut.result()
                total_imgs += imgs
                total_lbls += lbls
                print(f"  {name}: {imgs} images, {lbls} label files")

        print(f"  {split} total: {total_imgs} images, {total_lbls} label files → {output_dir}")


def main():
    """Parse args and run inspection or conversion."""
    parser = argparse.ArgumentParser(description="Convert Objects365 HF parquet to Ultralytics YOLO format")
    parser.add_argument("--input", required=True, type=Path, help="Directory with .parquet files")
    parser.add_argument("--output", type=Path, default=None, help="Output directory (YOLO structure)")
    parser.add_argument("--inspect", action="store_true", help="Print schema of first parquet and exit")
    parser.add_argument("--bbox-format", choices=["xyxy", "xywh"], default="xyxy",
                        help="Bbox format in parquet: xyxy=[x1,y1,x2,y2] or xywh=[x1,y1,w,h] (default: xyxy)")
    parser.add_argument("--cat-offset", type=int, default=0,
                        help="Subtract this from category IDs (use 1 if 1-indexed, 0 if already 0-indexed)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker processes (default: 4)")
    args = parser.parse_args()

    parquets = sorted(args.input.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No .parquet files found in {args.input}")

    if args.inspect:
        inspect_schema(parquets[0])
        return

    if args.output is None:
        parser.error("--output is required for conversion")

    convert(args.input, args.output, args.bbox_format, args.cat_offset, args.workers)
    print(f"\nDone. Use Objects365.yaml with path: {args.output}")


if __name__ == "__main__":
    main()
