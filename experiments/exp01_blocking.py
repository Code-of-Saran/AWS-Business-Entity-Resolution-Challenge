"""Experiment 1: candidate recall / cost of the blocking families.

Indexes the FULL Source-2/3 corpus for a country and probes it with a random
sample of Source-1 entities, so recall is measured against the real index size
while staying cheap enough to sweep configurations.

Usage: python3 experiments/exp01_blocking.py --country India --sample 50000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import polars as pl

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "code", "business_entity_resolution", "src")
sys.path.insert(0, os.path.abspath(SRC))
import blocking as bk          # noqa: E402
import data_loader as dl       # noqa: E402
from prepare import prepared_path  # noqa: E402


def load(split: str, source: int, country: str) -> pl.DataFrame:
    return (pl.scan_parquet(prepared_path(split, source))
              .filter(pl.col("country") == country)
              .collect())


def run(country: str, sample: int, cfg: bk.BlockConfig, seed: int = 42) -> dict:
    t0 = time.time()
    a_full = load("train", 1, country)
    b = pl.concat([load("train", 2, country), load("train", 3, country)])
    n_a_full, n_b = a_full.height, b.height

    a = a_full.sample(n=min(sample, n_a_full), seed=seed) if sample else a_full
    a = a.with_row_index("idx")
    b = b.with_row_index("idx")

    vat = bk.build_vocab([a_full, b], "addr_toks")
    van = bk.build_vocab([a_full, b], "addr_nums")
    vmt = bk.build_vocab([a_full, b], "name_toks")
    t_vocab = time.time() - t0

    a = bk.prepare_side(a, vat, van, vmt, cfg)
    b = bk.prepare_side(b, vat, van, vmt, cfg)
    t_prep = time.time() - t0

    cand = bk.generate_candidates(a, b, cfg)
    t_block = time.time() - t0

    # ---- ground truth for the sampled entities ----
    gt = dl.read_ground_truth()
    a_ids = a.select("idx", "entity_id")
    gt = (gt.join(a_ids, left_on="source1_entity_id", right_on="entity_id",
                  how="inner")
            .select("idx", pl.col("matched_entity_ids").str.split(",").alias("m")))
    truth = (gt.explode("m").drop_nulls("m").filter(pl.col("m") != "")
               .join(b.select(b_idx="idx", m="entity_id"), on="m", how="inner")
               .select(a_idx="idx", b_idx="b_idx"))

    hit = truth.join(cand.select("a_idx", "b_idx"), on=["a_idx", "b_idx"],
                     how="semi")
    n_true, n_hit = truth.height, hit.height

    per_s1 = cand.group_by("a_idx").len().rename({"len": "n"})
    per_s1 = (a.select("idx").rename({"idx": "a_idx"})
                .join(per_s1, on="a_idx", how="left")
                .with_columns(pl.col("n").fill_null(0)))
    # fraction of S1 entities whose every true match survived blocking
    tcnt = truth.group_by("a_idx").len().rename({"len": "t"})
    hcnt = hit.group_by("a_idx").len().rename({"len": "h"})
    cov = (tcnt.join(hcnt, on="a_idx", how="left")
               .with_columns(pl.col("h").fill_null(0)))

    res = {
        "country": country, "n_s1_sampled": a.height, "n_s1_total": n_a_full,
        "n_b_indexed": n_b,
        "config": {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in cfg.__dict__.items() if k != "extra"},
        "n_true_pairs": n_true, "n_true_pairs_found": n_hit,
        "candidate_recall": round(n_hit / n_true, 5) if n_true else None,
        "full_coverage_pct": round(100.0 * float((cov["h"] == cov["t"]).mean()), 3),
        "n_candidate_pairs": cand.height,
        "cands_per_s1_mean": round(float(per_s1["n"].mean()), 2),
        "cands_per_s1_median": int(per_s1["n"].median()),
        "cands_per_s1_p95": int(per_s1["n"].quantile(0.95)),
        "cands_per_s1_max": int(per_s1["n"].max()),
        "s1_with_zero_candidates_pct": round(
            100.0 * float((per_s1["n"] == 0).mean()), 3),
        "reduction_ratio": float(cand.height) / (a.height * n_b),
        "secs_vocab": round(t_vocab, 1), "secs_prepare": round(t_prep, 1),
        "secs_block": round(t_block, 1),
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="India")
    ap.add_argument("--sample", type=int, default=50000)
    ap.add_argument("--max-block", type=int, default=100)
    ap.add_argument("--families", default="")
    ap.add_argument("--tag", default="baseline")
    args = ap.parse_args()

    cfg = bk.BlockConfig(max_block=args.max_block)
    if args.families:
        name2f = {v: k for k, v in bk.FAMILY_NAMES.items()}
        cfg.families = tuple(name2f[x] for x in args.families.split(","))

    res = run(args.country, args.sample, cfg)
    res["tag"] = args.tag
    print(json.dumps(res, indent=2))
    log = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "exp01_blocking_log.jsonl")
    with open(log, "a") as f:
        f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
