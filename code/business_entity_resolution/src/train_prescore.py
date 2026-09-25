"""Stage 2a: train the cheap candidate ranker.

The ranker's only job is to shrink ~120 blocked candidates per Source-1 entity to
the top K the expensive stage can afford, while giving up as little recall as
possible. It sees only the integer set-overlap features, so it can score hundreds
of millions of pairs.

It is trained on fold 0 exclusively; the matching model trains on folds 1-7 and is
validated on folds 8-9, so no entity used to fit the ranker is ever scored as
validation data.

Usage:  python3 src/train_prescore.py
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
import pipeline as pp        # noqa: E402
import prescore as ps        # noqa: E402

MODEL_DIR = os.path.join(dl.ARTIFACT_ROOT, "models")
RANKER_PATH = os.path.join(MODEL_DIR, "prescore_ranker.txt")
META_PATH = os.path.join(MODEL_DIR, "prescore_ranker.json")


def collect_training_rows(cfg: bk.BlockConfig, per_country: int,
                          chunk: int, seed: int = 11):
    Xs, ys = [], []
    stats = {}
    gt = dl.read_ground_truth()
    for country in pp.countries("train"):
        t0 = time.time()
        a_full = pp.load_prepared("train", 1, country)
        ix = pp.build_country_index("train", country, cfg, a_full=a_full)
        a_fold = a_full.filter(pl.col("entity_id").pipe(pp.fold_of)
                               .is_in(list(pp.FOLD_PRESCORE)))
        if a_fold.height > per_country:
            a_fold = a_fold.sample(n=per_country, seed=seed)
        a_fold = a_fold.with_row_index("idx")
        truth = pp.truth_pairs(a_fold, ix.b, gt)
        n_hit = 0
        for part in pp.chunks(a_fold, chunk):
            a = pp.prepare_a_chunk(part, ix)
            cand = bk.probe(a, ix.kb, ix.cfg)
            sc = ps.attach_cheap_features(cand, a, ix.b, ix.thr_at, ix.thr_an,
                                          ix.thr_mt)
            sc = sc.join(truth.with_columns(y=pl.lit(1, pl.Int8)),
                         on=["a_idx", "b_idx"], how="left").with_columns(
                             pl.col("y").fill_null(0))
            Xs.append(sc.select(ps.CHEAP_FEATURES).to_numpy().astype(np.float32))
            ys.append(sc["y"].to_numpy().astype(np.int8))
            n_hit += int(sc["y"].sum())
        stats[country] = {
            "n_s1": a_fold.height, "n_index": ix.b.height,
            "true_pairs": truth.height,
            "blocking_recall": round(n_hit / truth.height, 5) if truth.height else None,
            "pairs": int(sum(x.shape[0] for x in Xs)),
            "secs": round(time.time() - t0, 1)}
        print(f"[prescore] {country}: {json.dumps(stats[country])}", flush=True)
        del ix, a_full
    return np.concatenate(Xs), np.concatenate(ys), stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-country", type=int, default=45000)
    ap.add_argument("--chunk", type=int, default=25000)
    ap.add_argument("--max-block", type=int, default=60)
    args = ap.parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)

    cfg = bk.BlockConfig(n_addr_tok=5, n_name_tok=4, max_block=args.max_block)
    X, y, stats = collect_training_rows(cfg, args.per_country, args.chunk)
    print(f"[prescore] training rows={X.shape[0]:,} positives={int(y.sum()):,}")

    import lightgbm as lgb
    t0 = time.time()
    model = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.08,
                               num_leaves=95, min_child_samples=60,
                               subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.9, random_state=0,
                               n_jobs=10, verbose=-1)
    model.fit(X, y, feature_name=ps.CHEAP_FEATURES)
    model.booster_.save_model(RANKER_PATH)
    meta = {"features": ps.CHEAP_FEATURES, "n_rows": int(X.shape[0]),
            "n_pos": int(y.sum()), "fit_secs": round(time.time() - t0, 1),
            "blocking": {k: v for k, v in cfg.__dict__.items() if k != "extra"},
            "per_country": stats,
            "importance": dict(sorted(
                zip(ps.CHEAP_FEATURES,
                    [int(v) for v in model.feature_importances_]),
                key=lambda t: -t[1]))}
    meta["blocking"]["max_block_by_family"] = {
        str(k): v for k, v in cfg.max_block_by_family.items()}
    meta["blocking"]["families"] = list(cfg.families)
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[prescore] saved {RANKER_PATH} (fit {meta['fit_secs']}s)")


def load_ranker():
    """Load the saved ranker wrapped so ``predict_proba`` works as in training."""
    import lightgbm as lgb

    booster = lgb.Booster(model_file=RANKER_PATH)

    class _Wrap:
        def predict_proba(self, X):
            p = booster.predict(X, num_threads=10)
            return np.column_stack([1.0 - p, p])

    return _Wrap()


if __name__ == "__main__":
    main()
