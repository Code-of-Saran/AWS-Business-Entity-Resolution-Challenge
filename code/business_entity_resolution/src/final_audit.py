"""Final submission audit: verify the package independently of the official validator.

Re-derives every claim from the files on disk rather than from any pipeline
artefact, so a bug in the pipeline cannot also hide itself here. Checks the
structural rules the scorer enforces, plus the ones it does not: that the
predictions are a subset of the candidates, that no entity is missing, and that
the outputs actually contain predictions rather than being structurally valid but
empty.

Usage:  python3 src/final_audit.py [--zip ../../CodeTitans_submission.zip]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile

import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_loader as dl  # noqa: E402

ROOT = dl.PROJECT_ROOT
OUT = os.path.join(ROOT, "output")
PLACEHOLDER_MARKERS = ["[Your Team Name]", "[List all team members]", "[Date]",
                       "[your best validation score]", "[total]"]


def _read_pairs(path: str, col: str) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", quote_char=None, has_header=True,
                       schema={"source1_entity_id": pl.Utf8, col: pl.Utf8},
                       missing_utf8_is_empty_string=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="")
    args = ap.parse_args()

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append((name, ok, detail))

    # ---- 1 test source counts ----
    n_s1 = pl.scan_csv(os.path.join(dl.DATA_ROOT, "test", "test_source1.tsv"),
                       separator="\t", quote_char=None).select(pl.len()).collect().item()
    valid_targets = set()
    for s in (2, 3):
        ids = pl.read_csv(os.path.join(dl.DATA_ROOT, "test", f"test_source{s}.tsv"),
                          separator="\t", quote_char=None,
                          columns=["entity_id"])["entity_id"]
        valid_targets |= set(ids.to_list())
    print(f"1.  test_source1 rows                 : {n_s1:,}")
    print(f"    test_source2+3 distinct ids       : {len(valid_targets):,}")

    m = _read_pairs(os.path.join(OUT, "matching_results.tsv"), "matched_entity_ids")
    c = _read_pairs(os.path.join(OUT, "candidate_pairs.tsv"), "candidate_entity_ids")

    # ---- 2-5 row and content counts ----
    m_nonempty = int((m["matched_entity_ids"].str.strip_chars() != "").sum())
    c_nonempty = int((c["candidate_entity_ids"].str.strip_chars() != "").sum())
    print(f"2.  matching_results.tsv rows          : {m.height:,}")
    print(f"3.  non-empty match rows               : {m_nonempty:,} "
          f"({100*m_nonempty/max(m.height,1):.2f}%)")
    print(f"4.  candidate_pairs.tsv rows           : {c.height:,}")
    print(f"5.  non-empty candidate rows           : {c_nonempty:,} "
          f"({100*c_nonempty/max(c.height,1):.2f}%)")
    check("matching_results has one row per test S1 entity", m.height == n_s1,
          f"{m.height:,} vs {n_s1:,}")
    check("candidate_pairs has one row per test S1 entity", c.height == n_s1,
          f"{c.height:,} vs {n_s1:,}")
    check("matching_results contains actual predictions (not an empty stub)",
          m_nonempty > 0, f"{m_nonempty:,} non-empty rows")
    check("candidate_pairs contains actual candidates", c_nonempty > 0,
          f"{c_nonempty:,} non-empty rows")

    # ---- 6 predicted ids ----
    mex = (m.filter(pl.col("matched_entity_ids") != "")
            .select("source1_entity_id",
                    pl.col("matched_entity_ids").str.split(",").alias("mid"))
            .explode("mid"))
    cex = (c.filter(pl.col("candidate_entity_ids") != "")
            .select("source1_entity_id",
                    pl.col("candidate_entity_ids").str.split(",").alias("mid"))
            .explode("mid"))
    print(f"6.  predicted match ids (total)        : {mex.height:,}")
    print(f"    candidate ids (total)             : {cex.height:,}")
    print(f"    mean matches per entity           : {mex.height/max(m.height,1):.3f}")

    # ---- structural rules ----
    check("headers exact",
          m.columns == ["source1_entity_id", "matched_entity_ids"]
          and c.columns == ["source1_entity_id", "candidate_entity_ids"],
          f"{m.columns} / {c.columns}")
    check("no duplicate source1_entity_id rows",
          m["source1_entity_id"].n_unique() == m.height
          and c["source1_entity_id"].n_unique() == c.height,
          f"{m['source1_entity_id'].n_unique():,} unique of {m.height:,}")
    dup_in_list = (m.filter(pl.col("matched_entity_ids") != "")
                    .select(n=pl.col("matched_entity_ids").str.split(",").list.len(),
                            u=pl.col("matched_entity_ids").str.split(",")
                              .list.unique().list.len())
                    .filter(pl.col("n") != pl.col("u")).height)
    check("no duplicate ids within a match list", dup_in_list == 0,
          f"{dup_in_list} offending rows")
    bad_prefix = mex.filter(~pl.col("mid").str.starts_with("S2-")
                            & ~pl.col("mid").str.starts_with("S3-")).height
    check("matched ids are S2-/S3- only (no self-matches)", bad_prefix == 0,
          f"{bad_prefix} offending ids")
    unknown = mex.filter(~pl.col("mid").is_in(pl.Series(list(valid_targets)))).height
    check("every matched id exists in the test set", unknown == 0,
          f"{unknown} unknown ids")
    # subset property
    offenders = (mex.join(cex, on=["source1_entity_id", "mid"], how="anti").height)
    check("matching_results is a subset of candidate_pairs", offenders == 0,
          f"{offenders} matched ids absent from candidates")
    # tab-separated / utf-8
    with open(os.path.join(OUT, "matching_results.tsv"), "rb") as f:
        head = f.readline()
    check("file is TAB-separated UTF-8", b"\t" in head and b"," not in head,
          repr(head[:60]))

    # ---- 7-10 validation metrics, read from the recorded run ----
    tune = {}
    p = os.path.join(ROOT, "experiments", "exp06_decision_tuning.json")
    if os.path.isfile(p):
        tune = json.load(open(p)).get("final", {})
    print(f"7.  candidate recall (validation)      : "
          f"{json.load(open(p)).get('candidate_recall', float('nan')):.5f}"
          if tune else "7.  candidate recall              : n/a")
    print(f"8.  validation precision               : {tune.get('micro_precision', float('nan')):.5f}")
    print(f"9.  validation recall                  : {tune.get('micro_recall', float('nan')):.5f}")
    print(f"10. validation macro F0.5              : {tune.get('macro_f05', float('nan')):.5f}")
    print(f"    validation singleton accuracy      : {tune.get('singleton_accuracy', float('nan')):.5f}")

    # ---- per-country prediction distribution ----
    s1 = pl.read_csv(os.path.join(dl.DATA_ROOT, "test", "test_source1.tsv"),
                     separator="\t", quote_char=None,
                     columns=["entity_id", "country"])
    j = (s1.join(m.rename({"source1_entity_id": "entity_id"}), on="entity_id", how="left")
          .with_columns(n=pl.when(pl.col("matched_entity_ids").str.strip_chars() == "")
                          .then(0)
                          .otherwise(pl.col("matched_entity_ids").str.split(",").list.len())))
    print("\n    per-country predicted match distribution:")
    print("    " + f"{'country':<10}{'entities':>12}{'mean':>8}{'singleton%':>12}"
          f"{'1':>8}{'2':>8}{'3':>8}{'4':>8}{'5+':>8}")
    for r in j.group_by("country").agg(
            n_ent=pl.len(), mean=pl.col("n").mean(),
            z=(pl.col("n") == 0).mean(), o=(pl.col("n") == 1).mean(),
            t=(pl.col("n") == 2).mean(), th=(pl.col("n") == 3).mean(),
            fo=(pl.col("n") == 4).mean(), fp=(pl.col("n") >= 5).mean()
    ).sort("country").iter_rows():
        print(f"    {r[0]:<10}{r[1]:>12,}{r[2]:>8.3f}{100*r[3]:>11.2f}%"
              + "".join(f"{100*x:>7.1f}%" for x in r[4:]))

    # ---- 14 placeholders ----
    doc = os.path.join(ROOT, "Documentation_template.md")
    remaining = []
    if os.path.isfile(doc):
        text = open(doc).read()
        remaining = [mk for mk in PLACEHOLDER_MARKERS if mk in text]
        members_needed = "[TEAM MEMBERS TO BE PROVIDED]" in text
    else:
        members_needed = True
    check("no unfilled template placeholders remain", not remaining,
          f"{remaining}" if remaining else "none")
    print(f"\n14. unfilled placeholders              : "
          f"{remaining if remaining else 'none'}")
    print(f"15. team-member names still required   : {members_needed}")

    # ---- 12-13 zip ----
    if args.zip and os.path.isfile(args.zip):
        with zipfile.ZipFile(args.zip) as z:
            names = z.namelist()
        print(f"\n12. zip path                           : {args.zip} "
              f"({os.path.getsize(args.zip)/1e6:.1f} MB)")
        print("13. zip contents:")
        for n in sorted(names):
            print(f"      {n}")
        need = ["output/matching_results.tsv", "output/candidate_pairs.tsv",
                "Documentation_template.md",
                "code/business_entity_resolution/README.md",
                "code/business_entity_resolution/requirements.txt"]
        check("zip contains every required file",
              all(any(x.endswith(n) for x in names) for n in need), "")
        check("zip excludes __pycache__ and .pyc",
              not any("__pycache__" in x or x.endswith(".pyc") for x in names), "")
    else:
        print(f"\n12. zip path                           : (not built)")

    print("\n" + "=" * 78)
    failed = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    print("=" * 78)
    print(f"{len(checks)-len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
