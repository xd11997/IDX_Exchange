from __future__ import annotations
"""
Week 1 Deliverable - IDX Exchange Internship
CRMLS monthly CSV basic cleaner + merger.

What it does:
1) Reads one CSV file or all CSV files in a folder.
2) Detects duplicated ".1" columns.
3) Verifies whether each base column and its ".1" twin are identical.
4) Drops ".1" duplicates when they are truly the same.
5) Adds provenance columns (source_file, source_month).
6) Optionally merges all cleaned files into one output CSV.
7) Writes a validation report so you can inspect anything unusual.

Typical usage:
    python crmls_clean_merge.py --input-dir /path/to/raw_csvs --output-dir /path/to/cleaned

Or single file:
    python crmls_clean_merge.py --input-file CRMLSListing202402.csv --output-dir cleaned

Notes:
- If a base column and its ".1" twin differ, the script keeps both columns and flags the file.
- This script is meant for initial cleaning before deeper feature engineering.
"""
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from datetime import datetime

def infer_month_from_filename(path: Path) -> datetime | None:
    """
    Infer YYYY-MM from filenames like CRMLSListing202402.csv or 202401.csv.
    Returns a datetime object (first day of month) or None if not found.
    """
    name = path.stem
    m = re.search(r"(20\d{2})[._-]?([01]\d)", name)
    if m:
        year = int(m.group(1))
        month = int(m.group(2))
        return datetime(year, month, 1)
    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def compare_series_exact(a: pd.Series, b: pd.Series) -> bool:
    return a.reset_index(drop=True).equals(b.reset_index(drop=True))


def clean_one_file(path: Path) -> Tuple[pd.DataFrame, dict]:
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
        "dropped_duplicate_cols": "",
        "kept_duplicate_cols": "",
        "rows_clean": 0,
        "cols_clean": 0,
    }

    dup_pairs: List[Tuple[str, str]] = []
    for col in list(df.columns):
        if col.endswith(".1"):
            base = col[:-2]
            if base in df.columns:
                dup_pairs.append((base, col))

    report["duplicate_pairs_found"] = len(dup_pairs)

    cols_to_drop = []
    dropped_cols = []
    kept_cols = []

    for base, dup in dup_pairs:
        if compare_series_exact(df[base], df[dup]):
            report["duplicate_pairs_identical"] += 1
            cols_to_drop.append(dup)
            dropped_cols.append(dup)
        else:
            report["duplicate_pairs_different"] += 1
            kept_cols.append(dup)

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    df["source_file"] = path.name
    df["source_month"] = source_month

    report["dropped_duplicate_cols"] = str(dropped_cols)
    report["kept_duplicate_cols"] = str(kept_cols)
    report["rows_clean"] = int(df.shape[0])
    report["cols_clean"] = int(df.shape[1])

    return df, report


def process_inputs(input_file: Path | None, input_dir: Path | None, output_dir: Path) -> None:
    if input_file is not None:
        files = [input_file]
    else:
        files = sorted(input_dir.glob("*.csv"))

    files = [p for p in files if p.is_file()]

    if not files:
        raise FileNotFoundError("No CSV files found.")

    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_frames = []
    reports = []
    dates = []

    for path in files:
        cleaned_df, report = clean_one_file(path)
        cleaned_frames.append(cleaned_df)
        reports.append(report)

        dt = infer_month_from_filename(path)
        if dt:
            dates.append(dt)

    if dates:
        latest_date = max(dates)
        date_str = latest_date.strftime("%Y_%m")
    else:
        date_str = "unknown"

    report_df = pd.DataFrame(reports)
    report_path = output_dir / f"merge_report_{date_str}.csv"
    report_df.to_csv(report_path, index=False)

    merged_df = pd.concat(cleaned_frames, ignore_index=True, sort=False)
    merged_path = output_dir / f"merged_data_{date_str}.csv"
    merged_df.to_csv(merged_path, index=False)

    print(f"Processed {len(files)} file(s)")
    print(f"Rows before concat: {int(report_df['rows_clean'].sum())}")
    print(f"Rows after concat: {len(merged_df)}")
    print(f"Report saved to: {report_path}")
    print(f"Merged file saved to: {merged_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and merge CRMLS monthly CSV files.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-dir", type=Path, help="Folder containing CSV files to clean and merge.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output folder for cleaned files and reports.")
    args = parser.parse_args()

    process_inputs(args.input_file, args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()