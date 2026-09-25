"""Stage 6: score every candidate for a split and persist the scored pairs.

Runs the same funnel as training, in Source-1 chunks, and writes one Parquet
shard per chunk containing (a_entity_id, b_entity_id, is_s3, cheap_score, p).
Keeping scoring and decision-making in separate steps means the decision rule can
be re-tuned, and the two submission files regenerated, without re-scoring ~35M
pairs.

The persisted set is exactly the top-K candidate list the model ran inference
over, so ``candidate_pairs.tsv`` is generated from it directly and
``matching_results.tsv`` is a strict subset by construction.

Usage:  python3 src/inference.py --split test
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blocking as bk        # noqa: E402
import data_loader as dl     # noqa: E402
import features as ft        # noqa: E402
import pipeline as pp        # noqa: E402
from train_prescore import load_ranker  # noqa: E402

SCORE_DIR = os.path.join(dl.ARTIFACT_ROOT, "scored")
MODEL_DIR = os.path.join(dl.ARTIFACT_ROOT, "models")


def score_dir(split: str) -> str:
    return os.path.join(SCORE_DIR, split)


def run(split: str, k: int, chunk: int, max_block: int, matcher_name: str,
        folds: tuple | None = None, fresh: bool = True) -> dict:
    """Score every candidate of ``split``; optionally restrict to Source-1 folds."""
    out_dir = score_dir(split)
    if fresh and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    import joblib
    matcher = joblib.load(os.path.join(MODEL_DIR, f"{matcher_name}.joblib"))
    ranker = load_ranker()
    cfg = bk.BlockConfig(n_addr_tok=5, n_name_tok=4, max_block=max_block)

    stats, t_all = {}, time.time()
    for country in pp.countries(split):
        t0 = time.time()
        a_full = pp.load_prepared(split, 1, country)
        if folds is not None:
            a_full = a_full.filter(
                pp.fold_of(pl.col("entity_id")).is_in(list(folds)))
        ix = pp.build_country_index(split, country, cfg, a_full=a_full)
        t_index = time.time() - t0

        a_full = a_full.with_row_index("idx")
        n_pairs, n_shard = 0, 0
        for part in pp.chunks(a_full, chunk):
            a = pp.prepare_a_chunk(part, ix)
            cand = pp.candidates_for_chunk(part, ix, ranker, k)
            if cand.height == 0:
                continue
            full = pp.attach_text(cand, a, ix)
            X = ft.build_feature_matrix(full, ix.ctry_desc)
            p = matcher.predict_proba(X)[:, 1].astype(np.float32)
            (full.select("a_entity_id", "b_entity_id", "is_s3", "cheap_score")
                 .with_columns(p=pl.Series("p", p))
                 .write_parquet(os.path.join(
                     out_dir, f"{country}_{n_shard:05d}.parquet"),
                     compression="zstd", compression_level=3))
            n_pairs += full.height
            n_shard += 1
            del full, X, p
        stats[country] = {"n_s1": a_full.height, "n_index": ix.b.height,
                          "n_pairs": n_pairs,
                          "pairs_per_s1": round(n_pairs / max(a_full.height, 1), 2),
                          "shards": n_shard, "secs_index": round(t_index, 1),
                          "secs_total": round(time.time() - t0, 1)}
        print(f"[infer] {country}: {json.dumps(stats[country])}", flush=True)
        del ix, a_full

    info = {"split": split, "k": k, "max_block": max_block,
            "matcher": matcher_name, "folds": list(folds) if folds else None,
            "per_country": stats, "secs_total": round(time.time() - t_all, 1)}
    with open(os.path.join(out_dir, "_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[infer] done in {info['secs_total']}s -> {out_dir}")
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=25000)
    ap.add_argument("--max-block", type=int, default=60)
    ap.add_argument("--matcher", default="matcher")
    ap.add_argument("--folds", default="", help="comma list of Source-1 folds")
    args = ap.parse_args()
    folds = tuple(int(x) for x in args.folds.split(",")) if args.folds else None
    run(args.split, args.k, args.chunk, args.max_block, args.matcher, folds)


if __name__ == "__main__":
    main()
