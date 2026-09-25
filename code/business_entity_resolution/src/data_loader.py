"""Loading utilities for the Business Entity Resolution challenge.

All challenge files are tab-separated with exactly 4 columns (2 for ground truth).
A small number of rows contain literal double-quote characters, so CSV quoting must
be disabled (``quote_char=None``) or fields get silently merged.

Nothing in this module writes to the original dataset directory.
"""
from __future__ import annotations

import os
from typing import Iterable

import polars as pl

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]
GT_COLS = ["source1_entity_id", "matched_entity_ids"]

# Resolve the dataset root: <repo>/dataset by default, overridable for portability.
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DATA_ROOT = os.environ.get("BER_DATA_ROOT", os.path.join(PROJECT_ROOT, "dataset"))
ARTIFACT_ROOT = os.environ.get("BER_ARTIFACTS", os.path.join(PROJECT_ROOT, "artifacts"))


def source_path(split: str, source: int) -> str:
    return os.path.join(DATA_ROOT, split, f"{split}_source{source}.tsv")


def ground_truth_path() -> str:
    return os.path.join(DATA_ROOT, "train", "train_ground_truth.tsv")


def _read_tsv(path: str, columns: Iterable[str]) -> pl.DataFrame:
    return pl.read_csv(
        path,
        separator="\t",
        quote_char=None,
        has_header=True,
        schema={c: pl.Utf8 for c in columns},
        missing_utf8_is_empty_string=True,
    )


def read_source(split: str, source: int) -> pl.DataFrame:
    """Read one source file as a DataFrame of Utf8 columns (empty string, not null)."""
    return _read_tsv(source_path(split, source), SOURCE_COLS)


def scan_source(split: str, source: int) -> pl.LazyFrame:
    return pl.scan_csv(
        source_path(split, source),
        separator="\t",
        quote_char=None,
        has_header=True,
        schema={c: pl.Utf8 for c in SOURCE_COLS},
        missing_utf8_is_empty_string=True,
    )


def read_ground_truth() -> pl.DataFrame:
    return _read_tsv(ground_truth_path(), GT_COLS)


def scan_ground_truth() -> pl.LazyFrame:
    return pl.scan_csv(
        ground_truth_path(),
        separator="\t",
        quote_char=None,
        has_header=True,
        schema={c: pl.Utf8 for c in GT_COLS},
        missing_utf8_is_empty_string=True,
    )
