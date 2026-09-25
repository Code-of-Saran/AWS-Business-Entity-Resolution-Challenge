"""Experiment 3: blocking-config sweep + recall@K after cheap pre-scoring.

Data is loaded once and reused across configurations. For each config the script
reports raw blocking recall/cost and then the recall that survives keeping only
the top-K candidates per Source-1 entity -- the number that actually bounds the
matching model.
"""
from __future__ import annotations

import argparse, json, os, sys, time
import polars as pl

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "code", "business_entity_resolution", "src")
sys.path.insert(0, os.path.abspath(SRC))
import blocking as bk
import prescore as ps
import data_loader as dl
from prepare import prepared_path

KS = (8, 12, 16, 20, 24, 32)


def load(split, source, country):
    return (pl.scan_parquet(prepared_path(split, source))
              .filter(pl.col("country") == country).collect())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="India")
    ap.add_argument("--sample", type=int, default=50000)
    args = ap.parse_args()

    a_full = load("train", 1, args.country)
    b_raw = pl.concat([load("train", 2, args.country), load("train", 3, args.country)])
    a_raw = a_full.sample(n=min(args.sample, a_full.height), seed=42).with_row_index("idx")
    b_raw = b_raw.with_row_index("idx")
    print(f"{args.country}: S1 sample={a_raw.height:,} of {a_full.height:,}, "
          f"index={b_raw.height:,}")

    vat = bk.build_vocab([a_full, b_raw], "addr_toks")
    van = bk.build_vocab([a_full, b_raw], "addr_nums")
    vmt = bk.build_vocab([a_full, b_raw], "name_toks")

    gt = dl.read_ground_truth()
    truth_ids = (gt.join(a_raw.select("idx", "entity_id"),
                         left_on="source1_entity_id", right_on="entity_id", how="inner")
                   .select("idx", pl.col("matched_entity_ids").str.split(",").alias("m"))
                   .explode("m").drop_nulls("m").filter(pl.col("m") != ""))
    truth = (truth_ids.join(b_raw.select(b_idx="idx", m="entity_id"), on="m", how="inner")
                      .select(a_idx="idx", b_idx="b_idx"))
    n_true = truth.height
    print(f"true pairs in scope = {n_true:,}\n")

    configs = {
        "base_at4_mb100":  bk.BlockConfig(),
        "at5_mt4_mb100":   bk.BlockConfig(n_addr_tok=5, n_name_tok=4),
        "at5_mt4_mb200":   bk.BlockConfig(n_addr_tok=5, n_name_tok=4, max_block=200),
        "at6_mt4_mb200":   bk.BlockConfig(n_addr_tok=6, n_name_tok=4, max_block=200),
        "at5_mt4_mb60":    bk.BlockConfig(n_addr_tok=5, n_name_tok=4, max_block=60),
    }
    out = []
    for tag, cfg in configs.items():
        t0 = time.time()
        a = bk.prepare_side(a_raw, vat, van, vmt, cfg)
        b = bk.prepare_side(b_raw, vat, van, vmt, cfg)
        cand = bk.generate_candidates(a, b, cfg)
        t_block = time.time() - t0
        hit = truth.join(cand.select("a_idx", "b_idx"), on=["a_idx", "b_idx"], how="semi").height
        rec = {"tag": tag, "raw_recall": round(hit / n_true, 5),
               "raw_pairs": cand.height,
               "raw_per_s1": round(cand.height / a_raw.height, 1),
               "secs_block": round(t_block, 1)}

        t1 = time.time()
        sc = ps.attach_cheap_scores(cand, a, b)
        for k in KS:
            tk = ps.top_k(sc, k)
            h = truth.join(tk.select("a_idx", "b_idx"), on=["a_idx", "b_idx"], how="semi").height
            rec[f"recall@{k}"] = round(h / n_true, 5)
            rec[f"per_s1@{k}"] = round(tk.height / a_raw.height, 2)
        rec["secs_prescore"] = round(time.time() - t1, 1)
        out.append(rec)
        print(json.dumps(rec))
        del a, b, cand, sc

    log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp03_sweep_log.jsonl")
    with open(log, "a") as f:
        for r in out:
            f.write(json.dumps({"country": args.country, **r}) + "\n")

    print("\n{:<18} {:>8} {:>9} {:>8} ".format("config", "rawRec", "raw/S1", "blk_s")
          + " ".join(f"{'R@'+str(k):>8}" for k in KS))
    for r in out:
        print("{:<18} {:>8.4f} {:>9.1f} {:>8.1f} ".format(
            r["tag"], r["raw_recall"], r["raw_per_s1"], r["secs_block"])
            + " ".join(f"{r['recall@'+str(k)]:>8.4f}" for k in KS))


if __name__ == "__main__":
    main()
