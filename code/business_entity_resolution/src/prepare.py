"""Stage 1: normalise every source file once and cache it as Parquet.

Running the normalisation once and reusing the cache keeps every later
experiment (blocking sweeps, feature builds, validation folds) cheap. The cache
lives under <project>/artifacts and the original dataset is never written to.

Usage:  python3 src/prepare.py [--split train|test]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_loader as dl  # noqa: E402
import preprocessing as pp  # noqa: E402

KEEP = ["entity_id", "country", "business_name", "business_address",
        "name_norm", "addr_norm", "name_toks", "addr_toks", "addr_nums",
        "name_key", "name_core", "name_core_nospace", "name_nospace", "name_nonlatin", "addr_nonlatin"]


def prepared_path(split: str, source: int) -> str:
    return os.path.join(dl.ARTIFACT_ROOT, "prepared", f"{split}_source{source}.parquet")


def prepare(split: str, source: int) -> str:
    out = prepared_path(split, source)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.time()
    df = dl.read_source(split, source)
    df = pp.add_normalized_columns(df).select(KEEP)
    df.write_parquet(out, compression="zstd", compression_level=3)
    n = df.height
    del df
    print(f"[prepare] {split} source{source}: {n:,} rows -> {out} "
          f"({time.time() - t0:.1f}s, {os.path.getsize(out) / 1e6:.0f} MB)",
          flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test", "both"], default="both")
    args = ap.parse_args()
    splits = ["train", "test"] if args.split == "both" else [args.split]
    for split in splits:
        for source in (1, 2, 3):
            prepare(split, source)


if __name__ == "__main__":
    main()
