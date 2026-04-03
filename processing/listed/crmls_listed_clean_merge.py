#!/usr/bin/env python3
"""
CRMLS monthly CSV cleaner + merger.

What it does:
1) Reads one CSV file or all CSV files in a folder.
2) Detects duplicated ".1" columns.
3) Verifies whether each base column and its ".1" twin are identical.
4) Drops ".1" duplicates when they are truly the same.
5) Standardizes a few common fields.
6) Adds provenance columns (source_file, source_month).
7) Optionally merges all cleaned files into one output CSV.
8) Writes a validation report so you can inspect anything unusual.

Typical usage:
    python crmls_clean_merge.py --input-dir /path/to/raw_csvs --output-dir /path/to/cleaned

Or single file:
    python crmls_clean_merge.py --input-file CRMLSListing202402.csv --output-dir cleaned

Notes:
- If a base column and its ".1" twin differ, the script keeps both columns and flags the file.
- This script is meant for initial cleaning before deeper feature engineering.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DATE_HINTS = ("date", "time", "timestamp")


def infer_month_from_filename(path: Path) -> str:
    """
    Try to infer YYYY-MM from filenames like CRMLSListing202402.csv or 202401.csv.
    Returns an empty string if not found.
    """
    name = path.stem
    m = re.search(r"(20\d{2})[._-]?([01]\d)", name)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return ""


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace and make column names unique-preserving."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def compare_series_exact(a: pd.Series, b: pd.Series) -> bool:
    """
    Compare two series with NaNs treated as equal and index alignment enforced.
    """
    if len(a) != len(b):
        return False
    # Same values + same missingness + same order
    return a.reset_index(drop=True).equals(b.reset_index(drop=True))


def safe_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse obvious date-like columns to datetime, but only when parsing succeeds
    for at least one non-null value. Unparseable values remain NaT.
    """
    df = df.copy()
    for col in df.columns:
        if any(h in col.lower() for h in DATE_HINTS):
            # Avoid over-parsing obviously numeric fields
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                non_null_before = df[col].notna().sum()
                non_null_after = parsed.notna().sum()
                # Accept parsing if it recovers at least one meaningful value
                if non_null_before > 0 and non_null_after > 0:
                    df[col] = parsed
            except Exception:
                pass
    return df


def add_source_columns(df: pd.DataFrame, source_file: str, source_month: str) -> pd.DataFrame:
    df = df.copy()
    df["source_file"] = source_file
    df["source_month"] = source_month
    return df


def clean_one_file(path: Path, output_dir: Path) -> Tuple[pd.DataFrame, Dict]:
    """
    Clean one CSV and write a cleaned version plus a validation report.
    """
    df = pd.read_csv(path, low_memory=False)
    df = normalize_columns(df)

    source_month = infer_month_from_filename(path)
    report = {
        "file": path.name,
        "rows_raw": int(df.shape[0]),
        "cols_raw": int(df.shape[1]),
        "source_month": source_month,
        "duplicate_pairs_found": 0,
        "duplicate_pairs_identical": 0,
        "duplicate_pairs_different": 0,
        "dropped_duplicate_cols": [],
        "kept_duplicate_cols": [],
    }

    # Detect base + ".1" twin columns
    dup_pairs: List[Tuple[str, str]] = []
    for col in list(df.columns):
        if col.endswith(".1"):
            base = col[:-2]
            if base in df.columns:
                dup_pairs.append((base, col))

    report["duplicate_pairs_found"] = len(dup_pairs)

    cols_to_drop = []
    for base, dup in dup_pairs:
        if compare_series_exact(df[base], df[dup]):
            report["duplicate_pairs_identical"] += 1
            cols_to_drop.append(dup)
            report["dropped_duplicate_cols"].append(dup)
        else:
            report["duplicate_pairs_different"] += 1
            report["kept_duplicate_cols"].append(dup)

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # ② 日期处理
    date_cols = [
    'CloseDate', 'ListingContractDate', 'PurchaseContractDate']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # ③ 数值处理
    numeric_cols = [
    'ListPrice', 'ClosePrice', 'LivingArea',
    'LotSizeSquareFeet', 'BathroomsTotalInteger',
    'BedroomsTotal', 'DaysOnMarket']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Standardize common types
    df = safe_parse_dates(df)

    # Add provenance
    df = add_source_columns(df, path.name, source_month)

    # Write cleaned file

    report["rows_clean"] = int(df.shape[0])
    report["cols_clean"] = int(df.shape[1])
    return df, report


def process_inputs(input_file: Path | None, input_dir: Path | None, output_dir: Path) -> None:
    if input_file is None and input_dir is None:
        raise ValueError("Provide either --input-file or --input-dir")

    files: List[Path] = []
    if input_file is not None:
        files = [input_file]
    else:
        files = sorted([p for p in input_dir.glob("*.csv") if p.is_file()])

    if not files:
        raise FileNotFoundError("No CSV files found.")

    all_cleaned = []
    reports = []

    for p in files:
        cleaned_df, report = clean_one_file(p, output_dir)
        all_cleaned.append(cleaned_df)
        reports.append(report)

    report_df = pd.DataFrame(reports)
    report_path = output_dir / "cleaning_report.csv"
    report_df.to_csv(report_path, index=False)

    # Merge only when processing multiple files or when the caller uses --input-dir.
    if len(all_cleaned) >= 1:
        merged = pd.concat(all_cleaned, ignore_index=True, sort=False)
        merged_path = output_dir / "merged_cleaned.csv"
        merged.to_csv(merged_path, index=False)

        # Also save a tiny JSON summary for quick inspection.
        summary = {
            "files_processed": len(files),
            "rows_merged": int(merged.shape[0]),
            "cols_merged": int(merged.shape[1]),
            "report_csv": report_path.name,
            "merged_csv": merged_path.name,
        }
        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(files)} file(s).")
    print(f"Output directory: {output_dir}")
    print(f"Report: {report_path}")
    print(f"Merged file: {output_dir / 'merged_cleaned.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Clean and merge CRMLS monthly CSV files.")
    parser.add_argument("--input-file", type=Path, help="Single CSV file to clean.")
    parser.add_argument("--input-dir", type=Path, help="Folder containing many CSV files to clean and merge.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output folder for cleaned files and reports.")
    args = parser.parse_args()

    process_inputs(args.input_file, args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
