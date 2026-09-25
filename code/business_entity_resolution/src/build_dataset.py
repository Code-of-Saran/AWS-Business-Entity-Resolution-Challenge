"""Stage 3a: materialise the pairwise training/validation dataset.

For every training country the full funnel is run over a sample of Source-1
entities drawn from the model folds (1-7) and the validation folds (8-9). The
surviving top-K candidates, their features and their labels are written to disk so
model/threshold experiments can iterate without re-running blocking.

Negative sampling note: the negatives here are exactly the candidates the model
will face at inference time -- they survived blocking *and* the cheap ranker, so
they are hard by construction (same street, same city, same name stem, or an
identical name at a different address). Training on this distribution rather than
on resampled random negatives keeps the training and inference distributions
identical, which is what lets a single probability threshold transfer.

Usage:  python3 src/build_dataset.py --train-per-country 150000 --valid-per-country 75000
"""
from __future__ import annotations

import argparse
import json
import os
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

DATASET_DIR = os.path.join(dl.ARTIFACT_ROOT, "dataset")
META_COLS = ["a_idx", "b_idx", "a_entity_id", "b_entity_id", "country",
             "is_s3", "cheap_score", "y", "split"]


def _build_for_country(country: str, cfg: bk.BlockConfig, ranker, k: int,
                       n_train: int, n_valid: int, chunk: int, seed: int,
                       gt: pl.DataFrame):
    a_full = pp.load_prepared("train", 1, country)
    ix = pp.build_country_index("train", country, cfg, a_full=a_full)

    fold = pp.fold_of(pl.col("entity_id"))
    plans = [("train", list(pp.FOLDS_TRAIN), n_train),
             ("valid", list(pp.FOLDS_VALID), n_valid)]
    X_parts, meta_parts, stats = [], [], {}
    for split_name, folds, n_want in plans:
        a_sel = a_full.filter(fold.is_in(folds))
        if n_want and a_sel.height > n_want:
            a_sel = a_sel.sample(n=n_want, seed=seed)
        a_sel = a_sel.with_row_index("idx")
        truth = pp.truth_pairs(a_sel, ix.b, gt)
        tcounts = pp.truth_counts(a_sel, gt)

        t0, n_pairs, n_hit = time.time(), 0, 0
        for part in pp.chunks(a_sel, chunk):
            a = pp.prepare_a_chunk(part, ix)
            cand = pp.candidates_for_chunk(part, ix, ranker, k)
            if cand.height == 0:
                continue
            full = pp.attach_text(cand, a, ix)
            full = full.join(truth.with_columns(y=pl.lit(1, pl.Int8)),
                             on=["a_idx", "b_idx"], how="left").with_columns(
                                 pl.col("y").fill_null(0))
            X_parts.append(ft.build_feature_matrix(full, ix.ctry_desc))
            meta_parts.append(full.select(
                "a_idx", "b_idx", "a_entity_id", "b_entity_id", "country",
                "is_s3", "cheap_score", "y",
                pl.lit(split_name).alias("split")))
            n_pairs += full.height
            n_hit += int(full["y"].sum())
        stats[split_name] = {
            "n_s1": a_sel.height, "true_pairs": truth.height,
            "true_pairs_all": int(tcounts["n_true"].sum()),
            "candidate_recall_vs_indexed": round(n_hit / truth.height, 5)
                if truth.height else None,
            "candidate_recall_vs_all": round(n_hit / int(tcounts["n_true"].sum()), 5)
                if int(tcounts["n_true"].sum()) else None,
            "pairs": n_pairs, "pairs_per_s1": round(n_pairs / a_sel.height, 2),
            "secs": round(time.time() - t0, 1)}
        # Per-entity truth counts are saved separately: entities whose true
        # matches never entered the candidate set still count in the macro mean.
        tcounts.with_columns(
            country=pl.lit(country), split=pl.lit(split_name)).write_parquet(
            os.path.join(DATASET_DIR, f"truth_{country}_{split_name}.parquet"))
        print(f"[dataset] {country}/{split_name}: {json.dumps(stats[split_name])}",
              flush=True)
    del ix, a_full
    return X_parts, meta_parts, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-per-country", type=int, default=150000)
    ap.add_argument("--valid-per-country", type=int, default=75000)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=25000)
    ap.add_argument("--max-block", type=int, default=60)
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()
    os.makedirs(DATASET_DIR, exist_ok=True)

    cfg = bk.BlockConfig(n_addr_tok=5, n_name_tok=4, max_block=args.max_block)
    ranker = load_ranker()
    gt = dl.read_ground_truth()

    all_X, all_meta, all_stats = [], [], {}
    for country in pp.countries("train"):
        Xp, Mp, st = _build_for_country(
            country, cfg, ranker, args.k, args.train_per_country,
            args.valid_per_country, args.chunk, args.seed, gt)
        all_X.extend(Xp)
        all_meta.extend(Mp)
        all_stats[country] = st

    X = np.concatenate(all_X)
    del all_X
    meta = pl.concat(all_meta)
    del all_meta
    np.save(os.path.join(DATASET_DIR, "X.npy"), X)
    meta.write_parquet(os.path.join(DATASET_DIR, "meta.parquet"))
    info = {"n_rows": int(X.shape[0]), "n_features": int(X.shape[1]),
            "features": ft.FEATURE_NAMES, "k": args.k,
            "blocking": {"n_addr_tok": cfg.n_addr_tok, "n_name_tok": cfg.n_name_tok,
                         "n_addr_num": cfg.n_addr_num, "max_block": cfg.max_block,
                         "max_block_by_family": {str(a): b for a, b in
                                                 cfg.max_block_by_family.items()},
                         "families": list(cfg.families)},
            "seed": args.seed, "per_country": all_stats}
    with open(os.path.join(DATASET_DIR, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[dataset] X={X.shape} positives={int(meta['y'].sum()):,} -> {DATASET_DIR}")


if __name__ == "__main__":
    main()
