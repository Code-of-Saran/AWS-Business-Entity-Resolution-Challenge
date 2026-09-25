"""Exploratory data analysis for the Business Entity Resolution challenge.

Writes a JSON report to <project>/reports/eda_report.json.
Read-only with respect to the dataset.

Usage:  python3 src/eda.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import unicodedata

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_loader as dl  # noqa: E402

REPORT_DIR = os.path.join(dl.PROJECT_ROOT, "reports")
SPLITS = {"train": (1, 2, 3), "test": (1, 2, 3)}

_SCRIPT_TAGS = (
    "DEVANAGARI", "TAMIL", "TELUGU", "BENGALI", "GUJARATI", "KANNADA",
    "MALAYALAM", "GURMUKHI", "ORIYA", "ARABIC", "CYRILLIC", "GREEK",
    "HAN", "HIRAGANA", "KATAKANA", "HANGUL",
)


def _script_of(text: str) -> str:
    """Coarse writing-system label for a string, from its first cased letter.

    Used only to report how much of each source is transliterated; no model
    component branches on this.
    """
    for ch in text:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        for tag in _SCRIPT_TAGS:
            if tag in name:
                return tag
        return "LATIN"
    return "NONE"


def profile_source(split: str, source: int) -> dict:
    t0 = time.time()
    df = dl.read_source(split, source)
    n = df.height
    out: dict = {"split": split, "source": source, "n_rows": n,
                 "columns": df.columns}

    miss = {}
    for c in dl.SOURCE_COLS:
        n_empty = int((df[c].str.strip_chars() == "").sum())
        miss[c] = {"empty_or_whitespace": n_empty,
                   "empty_pct": round(100.0 * n_empty / n, 4)}
    out["missing"] = miss

    out["id"] = {
        "n_unique": int(df["entity_id"].n_unique()),
        "prefixes": {k: int(v) for k, v in
                     df["entity_id"].str.slice(0, 3).value_counts().iter_rows()},
        "numeric_min": int(df["entity_id"].str.slice(3).cast(pl.Int64).min()),
        "numeric_max": int(df["entity_id"].str.slice(3).cast(pl.Int64).max()),
    }

    dup = {
        "duplicate_entity_ids": n - out["id"]["n_unique"],
        "distinct_names": int(df["business_name"].n_unique()),
        "distinct_addresses": int(df["business_address"].n_unique()),
        "distinct_name_address": int(
            df.select(pl.concat_str(["business_name", "business_address"],
                                    separator="|#|")).to_series().n_unique()),
    }
    dup["rows_sharing_a_name"] = n - dup["distinct_names"]
    dup["rows_sharing_name_address"] = n - dup["distinct_name_address"]
    out["duplicates"] = dup

    out["country"] = {(k if k else "<EMPTY>"): int(v) for k, v in
                      df["country"].value_counts(sort=True).iter_rows()}

    lens = {}
    for c in ["business_name", "business_address"]:
        L = df[c].str.len_chars()
        toks = df[c].str.strip_chars().str.split(" ").list.len()
        lens[c] = {
            "mean": round(float(L.mean()), 2),
            "p50": int(L.quantile(0.5)), "p90": int(L.quantile(0.9)),
            "p99": int(L.quantile(0.99)), "max": int(L.max()),
            "tokens_mean": round(float(toks.mean()), 2),
            "tokens_p90": int(toks.quantile(0.9)),
        }
    out["lengths"] = lens

    sample = df["business_name"].sample(n=min(200_000, n), seed=13).to_list()
    scripts: dict[str, int] = {}
    for s in sample:
        k = _script_of(s)
        scripts[k] = scripts.get(k, 0) + 1
    out["name_script_sample_pct"] = {
        k: round(100.0 * v / len(sample), 3)
        for k, v in sorted(scripts.items(), key=lambda kv: -kv[1])}

    out["address_has_digit_pct"] = round(
        100.0 * float(df["business_address"].str.contains(r"[0-9]").mean()), 3)
    out["name_has_digit_pct"] = round(
        100.0 * float(df["business_name"].str.contains(r"[0-9]").mean()), 3)

    out["elapsed_s"] = round(time.time() - t0, 1)
    del df
    return out


def profile_ground_truth(s1_country: pl.DataFrame) -> dict:
    t0 = time.time()
    gt = dl.read_ground_truth()
    n = gt.height
    is_empty = pl.col("matched_entity_ids").str.strip_chars() == ""
    lst = pl.col("matched_entity_ids").str.split(",")
    gt = gt.with_columns(
        n_matches=pl.when(is_empty).then(0).otherwise(lst.list.len()),
        n_s2=pl.when(is_empty).then(0).otherwise(
            lst.list.eval(pl.element().str.starts_with("S2-")).list.sum()),
    )
    gt = gt.with_columns(n_s3=pl.col("n_matches") - pl.col("n_s2"))

    out: dict = {"n_rows": n, "n_unique_s1": int(gt["source1_entity_id"].n_unique())}
    nm = gt["n_matches"]
    out["singletons"] = int((nm == 0).sum())
    out["singleton_pct"] = round(100.0 * out["singletons"] / n, 3)
    out["total_positive_pairs"] = int(nm.sum())
    out["matches_per_s1"] = {
        "mean": round(float(nm.mean()), 4),
        "mean_nonsingleton": round(float(nm.filter(nm > 0).mean()), 4),
        "p50": int(nm.quantile(0.5)), "p90": int(nm.quantile(0.9)),
        "p99": int(nm.quantile(0.99)), "max": int(nm.max()),
    }
    out["match_count_distribution"] = {
        int(k): int(v) for k, v in
        nm.value_counts().sort("n_matches").iter_rows()}
    out["s2_matches_total"] = int(gt["n_s2"].sum())
    out["s3_matches_total"] = int(gt["n_s3"].sum())
    out["per_s1_source_pattern"] = {
        k: int(v) for k, v in
        gt.select(pattern=pl.when((pl.col("n_s2") > 0) & (pl.col("n_s3") > 0))
                  .then(pl.lit("S2+S3"))
                  .when(pl.col("n_s2") > 0).then(pl.lit("S2_only"))
                  .when(pl.col("n_s3") > 0).then(pl.lit("S3_only"))
                  .otherwise(pl.lit("none")))["pattern"]
        .value_counts(sort=True).iter_rows()}
    out["s2_per_s1"] = {"p50": int(gt["n_s2"].quantile(0.5)),
                        "p90": int(gt["n_s2"].quantile(0.9)),
                        "max": int(gt["n_s2"].max())}
    out["s3_per_s1"] = {"p50": int(gt["n_s3"].quantile(0.5)),
                        "p90": int(gt["n_s3"].quantile(0.9)),
                        "max": int(gt["n_s3"].max())}

    j = gt.join(s1_country, left_on="source1_entity_id", right_on="entity_id",
                how="left")
    out["gt_rows_with_no_matching_s1_record"] = int(j["country"].is_null().sum())
    per_country = (
        j.group_by("country")
        .agg(n_s1=pl.len(),
             singletons=(pl.col("n_matches") == 0).sum(),
             mean_matches=pl.col("n_matches").mean(),
             mean_matches_nonsingleton=pl.col("n_matches")
                 .filter(pl.col("n_matches") > 0).mean(),
             total_pairs=pl.col("n_matches").sum(),
             s2_pairs=pl.col("n_s2").sum(), s3_pairs=pl.col("n_s3").sum())
        .sort("n_s1", descending=True))
    out["per_country"] = [
        {"country": r[0], "n_s1": r[1], "singletons": r[2],
         "singleton_pct": round(100.0 * r[2] / r[1], 3),
         "mean_matches": round(float(r[3]), 4),
         "mean_matches_nonsingleton": round(float(r[4] or 0.0), 4),
         "total_pairs": r[5], "s2_pairs": r[6], "s3_pairs": r[7]}
        for r in per_country.iter_rows()]

    expl = (gt.filter(pl.col("n_matches") > 0)
              .select(pl.col("matched_entity_ids").str.split(",").alias("m"))
              .explode("m"))
    out["distinct_matched_ids"] = int(expl["m"].n_unique())
    out["matched_id_rows"] = expl.height
    out["matched_ids_reused_across_s1"] = (
        out["matched_id_rows"] - out["distinct_matched_ids"])
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def main() -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)
    report: dict = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "sources": []}
    for split, sources in SPLITS.items():
        for src in sources:
            prof = profile_source(split, src)
            report["sources"].append(prof)
            print(f"[eda] {split} source{src}: {prof['n_rows']:,} rows "
                  f"({prof['elapsed_s']}s)", flush=True)

    s1_country = dl.read_source("train", 1).select(["entity_id", "country"])
    report["ground_truth"] = profile_ground_truth(s1_country)
    print(f"[eda] ground truth done ({report['ground_truth']['elapsed_s']}s)",
          flush=True)

    path = os.path.join(REPORT_DIR, "eda_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[eda] wrote {path}")


if __name__ == "__main__":
    main()
