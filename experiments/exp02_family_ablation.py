"""Experiment 2: per-family recall / cost decomposition, plus missed-pair analysis.

One blocking pass is run with every family enabled; the returned ``fam_mask``
then lets each family's individual contribution be read off without re-blocking.
"""
from __future__ import annotations

import argparse, json, os, sys, time
import polars as pl

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "code", "business_entity_resolution", "src")
sys.path.insert(0, os.path.abspath(SRC))
import blocking as bk
import data_loader as dl
from prepare import prepared_path


def load(split, source, country):
    return (pl.scan_parquet(prepared_path(split, source))
              .filter(pl.col("country") == country).collect())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="India")
    ap.add_argument("--sample", type=int, default=50000)
    ap.add_argument("--max-block", type=int, default=100)
    ap.add_argument("--show-missed", type=int, default=12)
    args = ap.parse_args()

    cfg = bk.BlockConfig(max_block=args.max_block)
    a_full = load("train", 1, args.country)
    b = pl.concat([load("train", 2, args.country), load("train", 3, args.country)])
    a = a_full.sample(n=min(args.sample, a_full.height), seed=42).with_row_index("idx")
    b = b.with_row_index("idx")

    vat = bk.build_vocab([a_full, b], "addr_toks")
    van = bk.build_vocab([a_full, b], "addr_nums")
    vmt = bk.build_vocab([a_full, b], "name_toks")
    ap_ = bk.prepare_side(a, vat, van, vmt, cfg)
    bp = bk.prepare_side(b, vat, van, vmt, cfg)
    t0 = time.time()
    cand = bk.generate_candidates(ap_, bp, cfg)
    print(f"blocking {time.time()-t0:.1f}s  candidates={cand.height:,}")

    gt = dl.read_ground_truth()
    truth = (gt.join(a.select("idx", "entity_id"),
                     left_on="source1_entity_id", right_on="entity_id", how="inner")
               .select("idx", pl.col("matched_entity_ids").str.split(",").alias("m"))
               .explode("m").drop_nulls("m").filter(pl.col("m") != "")
               .join(b.select(b_idx="idx", m="entity_id"), on="m", how="inner")
               .select(a_idx="idx", b_idx="b_idx"))
    n_true = truth.height
    print(f"true pairs = {n_true:,}\n")

    def stats(c, label):
        h = truth.join(c.select("a_idx", "b_idx"), on=["a_idx", "b_idx"], how="semi").height
        per = c.group_by("a_idx").len()["len"]
        print(f"  {label:<26} recall={h/n_true:7.4%}  pairs={c.height:>10,}  "
              f"mean/S1={c.height/a.height:7.1f}  p95={int(per.quantile(0.95)) if per.len() else 0}")
        return h / n_true, c.height

    print("per-family (alone):")
    rows = []
    for f, nm in bk.FAMILY_NAMES.items():
        sel = cand.filter((pl.col("fam_mask") // (2 ** f)) % 2 == 1)
        r, n = stats(sel, nm)
        rows.append((nm, f, r, n))

    print("\ngreedy cumulative (by marginal recall gain):")
    chosen, mask_expr = [], None
    remaining = set(bk.FAMILY_NAMES)
    while remaining:
        best = None
        for f in remaining:
            e = (pl.col("fam_mask") // (2 ** f)) % 2 == 1
            cur = e if mask_expr is None else (mask_expr | e)
            h = truth.join(cand.filter(cur).select("a_idx", "b_idx"),
                           on=["a_idx", "b_idx"], how="semi").height
            if best is None or h > best[1]:
                best = (f, h, cur)
        f, h, cur = best
        mask_expr = cur
        chosen.append(bk.FAMILY_NAMES[f])
        remaining.discard(f)
        c = cand.filter(cur)
        print(f"  +{bk.FAMILY_NAMES[f]:<4} -> recall={h/n_true:7.4%}  "
              f"pairs={c.height:>10,}  mean/S1={c.height/a.height:7.1f}   [{'+'.join(chosen)}]")

    # ---- what is missed? ----
    missed = truth.join(cand.select("a_idx", "b_idx"), on=["a_idx", "b_idx"], how="anti")
    print(f"\nmissed true pairs: {missed.height:,} ({missed.height/n_true:.3%})")
    m = (missed.sample(n=min(args.show_missed, missed.height), seed=3)
               .join(a.select("a_idx=idx", *[]) if False else
                     a.select(pl.col("idx").alias("a_idx"), pl.col("business_name").alias("an"),
                              pl.col("business_address").alias("aa")), on="a_idx")
               .join(b.select(pl.col("idx").alias("b_idx"), pl.col("business_name").alias("bn"),
                              pl.col("business_address").alias("ba"),
                              pl.col("entity_id").alias("bid")), on="b_idx"))
    pl.Config.set_fmt_str_lengths(160)
    for r in m.iter_rows(named=True):
        print("-" * 96)
        print(f"  S1 NAME: {r['an']}\n  S1 ADDR: {r['aa']}")
        print(f"  {r['bid']} NAME: {r['bn']}\n  {'':>13} ADDR: {r['ba']}")


if __name__ == "__main__":
    main()
