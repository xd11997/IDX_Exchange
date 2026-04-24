from __future__ import annotations

"""
Week 4-5 cleaning pipeline.

What this script does and why:
1. Standardizes blank strings to missing values so downstream summaries are honest.
2. Converts required date fields to datetime for reliable time logic checks.
3. Converts core analysis fields to numeric so invalid values can be detected.
4. Drops a small set of high-cardinality contact/detail columns that are not needed
   for the analysis-ready dataset.
5. Flags impossible numeric values, removes those invalid rows, and reports counts.
6. Flags date-order violations without discarding those rows automatically.
7. Flags geographic quality issues without discarding those rows automatically.
8. Saves both the cleaned CSV and a JSON summary documenting every transformation.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_SOLD_INPUT_PATH = PROJECT_ROOT / "processed/sold/merged_sold_with_rate.csv"
DEFAULT_LISTED_INPUT_PATH = PROJECT_ROOT / "processed/listed/merged_listed_with_rate.csv"
DEFAULT_SOLD_OUTPUT_PATH = PROJECT_ROOT / "processed/sold/sold_analysis_ready_week45.csv"
DEFAULT_LISTED_OUTPUT_PATH = PROJECT_ROOT / "processed/listed/listed_analysis_ready_week45.csv"
DEFAULT_SOLD_REPORT_PATH = PROJECT_ROOT / "processed/sold/sold_analysis_ready_week45_summary.json"
DEFAULT_LISTED_REPORT_PATH = PROJECT_ROOT / "processed/listed/listed_analysis_ready_week45_summary.json"

DATE_COLUMNS = [
    "CloseDate",
    "PurchaseContractDate",
    "ListingContractDate",
    "ContractStatusChangeDate",
]

NUMERIC_COLUMNS = [
    "ClosePrice",
    "OriginalListPrice",
    "ListPrice",
    "LivingArea",
    "DaysOnMarket",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "Latitude",
    "Longitude",
]

DEFAULT_DROP_COLUMNS = [
    "ListAgentEmail",
    "ListAgentFirstName",
    "ListAgentLastName",
    "ListAgentFullName",
    "CoListAgentFirstName",
    "CoListAgentLastName",
    "BuyerAgentFirstName",
    "BuyerAgentLastName",
    "BuyerAgentMlsId",
    "BuyerAgentAOR",
    "ListAgentAOR",
    "BuyerOfficeName",
    "BuyerOfficeAOR",
    "CoListOfficeName",
    "ListOfficeName",
    "UnparsedAddress",
    "StreetNumberNumeric",
]

BOOLEAN_FLAG_COLUMNS = [
    "invalid_close_price_flag",
    "invalid_living_area_flag",
    "invalid_days_on_market_flag",
    "invalid_bedrooms_flag",
    "invalid_bathrooms_flag",
    "listing_after_close_flag",
    "purchase_after_close_flag",
    "negative_timeline_flag",
    "missing_coordinates_flag",
    "zero_coordinates_flag",
    "positive_longitude_flag",
    "implausible_coordinates_flag",
    "invalid_coordinates_flag",
]


def normalize_missing_strings(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Convert blank strings and common null markers into pandas missing values."""
    out = df.copy()
    summary: dict[str, int] = {}

    for col in out.columns:
        if not pd.api.types.is_object_dtype(out[col]) and not pd.api.types.is_string_dtype(out[col]):
            continue

        before_nulls = int(out[col].isna().sum())
        out[col] = out[col].replace(r"^\s*$", pd.NA, regex=True)
        out[col] = out[col].replace({"N/A": pd.NA, "NA": pd.NA, "None": pd.NA, "null": pd.NA})
        after_nulls = int(out[col].isna().sum())
        if after_nulls > before_nulls:
            summary[col] = after_nulls - before_nulls

    return out, summary


def convert_date_fields(
    df: pd.DataFrame,
    date_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Convert required date fields to datetime and record parse results."""
    date_columns = date_columns or DATE_COLUMNS
    out = df.copy()
    summary: dict[str, dict[str, Any]] = {}

    for col in date_columns:
        if col not in out.columns:
            summary[col] = {
                "present": False,
                "dtype_before": None,
                "dtype_after": None,
                "nulls_before": None,
                "nulls_after": None,
                "new_nulls_from_parse": None,
            }
            continue

        dtype_before = str(out[col].dtype)
        nulls_before = int(out[col].isna().sum())
        out[col] = pd.to_datetime(out[col], errors="coerce")
        nulls_after = int(out[col].isna().sum())

        summary[col] = {
            "present": True,
            "dtype_before": dtype_before,
            "dtype_after": str(out[col].dtype),
            "nulls_before": nulls_before,
            "nulls_after": nulls_after,
            "new_nulls_from_parse": nulls_after - nulls_before,
        }

    return out, summary


def ensure_numeric_fields(
    df: pd.DataFrame,
    numeric_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Coerce core analysis fields to numeric and record coercion results."""
    numeric_columns = numeric_columns or NUMERIC_COLUMNS
    out = df.copy()
    summary: dict[str, dict[str, Any]] = {}

    for col in numeric_columns:
        if col not in out.columns:
            summary[col] = {
                "present": False,
                "dtype_before": None,
                "dtype_after": None,
                "nulls_before": None,
                "nulls_after": None,
                "new_nulls_from_coerce": None,
            }
            continue

        dtype_before = str(out[col].dtype)
        nulls_before = int(out[col].isna().sum())
        out[col] = pd.to_numeric(out[col], errors="coerce")
        nulls_after = int(out[col].isna().sum())

        summary[col] = {
            "present": True,
            "dtype_before": dtype_before,
            "dtype_after": str(out[col].dtype),
            "nulls_before": nulls_before,
            "nulls_after": nulls_after,
            "new_nulls_from_coerce": nulls_after - nulls_before,
        }

    return out, summary


def drop_unnecessary_columns(
    df: pd.DataFrame,
    drop_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Remove a compact set of non-essential contact/detail columns."""
    drop_columns = drop_columns or DEFAULT_DROP_COLUMNS
    present_columns = [col for col in drop_columns if col in df.columns]
    return df.drop(columns=present_columns), present_columns


def flag_invalid_numeric_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Flag impossible numeric values required by the assignment."""
    out = df.copy()

    out["invalid_close_price_flag"] = out["ClosePrice"].le(0).fillna(False) if "ClosePrice" in out.columns else False
    out["invalid_living_area_flag"] = out["LivingArea"].le(0).fillna(False) if "LivingArea" in out.columns else False
    out["invalid_days_on_market_flag"] = (
        out["DaysOnMarket"].lt(0).fillna(False) if "DaysOnMarket" in out.columns else False
    )
    out["invalid_bedrooms_flag"] = (
        out["BedroomsTotal"].lt(0).fillna(False) if "BedroomsTotal" in out.columns else False
    )
    out["invalid_bathrooms_flag"] = (
        out["BathroomsTotalInteger"].lt(0).fillna(False)
        if "BathroomsTotalInteger" in out.columns
        else False
    )

    summary = {
        "invalid_close_price_flag": int(out["invalid_close_price_flag"].sum()),
        "invalid_living_area_flag": int(out["invalid_living_area_flag"].sum()),
        "invalid_days_on_market_flag": int(out["invalid_days_on_market_flag"].sum()),
        "invalid_bedrooms_flag": int(out["invalid_bedrooms_flag"].sum()),
        "invalid_bathrooms_flag": int(out["invalid_bathrooms_flag"].sum()),
    }
    return out, summary


def remove_invalid_numeric_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove rows with assignment-defined impossible numeric values."""
    invalid_mask = df[
        [
            "invalid_close_price_flag",
            "invalid_living_area_flag",
            "invalid_days_on_market_flag",
            "invalid_bedrooms_flag",
            "invalid_bathrooms_flag",
        ]
    ].any(axis=1)

    cleaned = df.loc[~invalid_mask].copy()
    summary = {
        "rows_before": int(len(df)),
        "rows_removed": int(invalid_mask.sum()),
        "rows_after": int(len(cleaned)),
    }
    return cleaned, summary


def flag_date_consistency(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Create the required date-order flags."""
    out = df.copy()

    listing = out["ListingContractDate"] if "ListingContractDate" in out.columns else pd.Series(pd.NaT, index=out.index)
    purchase = out["PurchaseContractDate"] if "PurchaseContractDate" in out.columns else pd.Series(pd.NaT, index=out.index)
    close = out["CloseDate"] if "CloseDate" in out.columns else pd.Series(pd.NaT, index=out.index)

    out["listing_after_close_flag"] = ((listing.notna()) & (close.notna()) & (listing > close)).fillna(False)
    out["purchase_after_close_flag"] = ((purchase.notna()) & (close.notna()) & (purchase > close)).fillna(False)
    out["negative_timeline_flag"] = ((listing.notna()) & (purchase.notna()) & (purchase < listing)).fillna(False)

    summary = {
        "listing_after_close_flag": int(out["listing_after_close_flag"].sum()),
        "purchase_after_close_flag": int(out["purchase_after_close_flag"].sum()),
        "negative_timeline_flag": int(out["negative_timeline_flag"].sum()),
    }
    return out, summary


def flag_geographic_quality(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Flag missing and implausible California coordinates."""
    out = df.copy()

    lat = out["Latitude"] if "Latitude" in out.columns else pd.Series(pd.NA, index=out.index, dtype="float64")
    lon = out["Longitude"] if "Longitude" in out.columns else pd.Series(pd.NA, index=out.index, dtype="float64")

    out["missing_coordinates_flag"] = lat.isna() | lon.isna()
    out["zero_coordinates_flag"] = lat.eq(0).fillna(False) | lon.eq(0).fillna(False)
    out["positive_longitude_flag"] = lon.gt(0).fillna(False)

    lat_outside_ca = ~(lat.between(32.0, 42.5))
    lon_outside_ca = ~(lon.between(-125.0, -113.0))
    out["implausible_coordinates_flag"] = (
        (~out["missing_coordinates_flag"])
        & (~out["zero_coordinates_flag"])
        & (lat_outside_ca | lon_outside_ca)
    ).fillna(False)

    out["invalid_coordinates_flag"] = (
        out["missing_coordinates_flag"]
        | out["zero_coordinates_flag"]
        | out["positive_longitude_flag"]
        | out["implausible_coordinates_flag"]
    )

    summary = {
        "missing_coordinates_flag": int(out["missing_coordinates_flag"].sum()),
        "zero_coordinates_flag": int(out["zero_coordinates_flag"].sum()),
        "positive_longitude_flag": int(out["positive_longitude_flag"].sum()),
        "implausible_coordinates_flag": int(out["implausible_coordinates_flag"].sum()),
        "invalid_coordinates_flag": int(out["invalid_coordinates_flag"].sum()),
    }
    return out, summary


def confirm_selected_dtypes(df: pd.DataFrame) -> dict[str, str]:
    """Capture final dtypes for key analysis fields and generated flags."""
    columns = [col for col in DATE_COLUMNS + NUMERIC_COLUMNS + BOOLEAN_FLAG_COLUMNS if col in df.columns]
    return {col: str(df[col].dtype) for col in columns}


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    return value


def print_section(title: str) -> None:
    print(f"\n{title}")


def run_pipeline(input_path: Path, output_path: Path, report_output: Path, dataset_name: str) -> None:
    print_section(f"{dataset_name.upper()} PIPELINE")
    print("Step 1/8: Loading dataset")
    df = pd.read_csv(input_path, low_memory=False)
    row_count_before = int(len(df))
    col_count_before = int(df.shape[1])
    print(f"Loaded {row_count_before:,} rows and {col_count_before:,} columns from {input_path}")

    print("Step 2/8: Standardizing missing values")
    df, missing_string_summary = normalize_missing_strings(df)
    print(f"Columns with new missing values from blank-string cleanup: {len(missing_string_summary)}")

    print("Step 3/8: Converting required date fields")
    df, date_summary = convert_date_fields(df)
    print("Date conversion complete")

    print("Step 4/8: Converting numeric analysis fields")
    df, numeric_summary = ensure_numeric_fields(df)
    print("Numeric coercion complete")

    print("Step 5/8: Dropping unnecessary columns")
    df, dropped_columns = drop_unnecessary_columns(df)
    print(f"Dropped {len(dropped_columns)} columns")

    print("Step 6/8: Flagging and removing invalid numeric values")
    df, invalid_numeric_summary = flag_invalid_numeric_values(df)
    df, invalid_row_summary = remove_invalid_numeric_rows(df)
    print(f"Removed {invalid_row_summary['rows_removed']:,} rows with impossible numeric values")

    print("Step 7/8: Running date consistency checks")
    df, date_flag_summary = flag_date_consistency(df)
    print(
        "Date flags:"
        f" listing_after_close={date_flag_summary['listing_after_close_flag']:,},"
        f" purchase_after_close={date_flag_summary['purchase_after_close_flag']:,},"
        f" negative_timeline={date_flag_summary['negative_timeline_flag']:,}"
    )

    print("Step 8/8: Running geographic quality checks")
    df, geo_summary = flag_geographic_quality(df)
    print(
        "Geographic flags:"
        f" missing={geo_summary['missing_coordinates_flag']:,},"
        f" zero={geo_summary['zero_coordinates_flag']:,},"
        f" positive_longitude={geo_summary['positive_longitude_flag']:,},"
        f" implausible={geo_summary['implausible_coordinates_flag']:,}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    report = {
        "dataset_name": dataset_name,
        "input_path": input_path,
        "output_path": output_path,
        "row_counts": {
            "before_cleaning": row_count_before,
            "after_cleaning": int(len(df)),
        },
        "column_counts": {
            "before_cleaning": col_count_before,
            "after_cleaning": int(df.shape[1]),
        },
        "blank_string_cleanup": missing_string_summary,
        "date_conversion_summary": date_summary,
        "numeric_conversion_summary": numeric_summary,
        "dropped_columns": dropped_columns,
        "invalid_numeric_flag_counts": invalid_numeric_summary,
        "invalid_numeric_row_removal": invalid_row_summary,
        "date_consistency_flag_counts": date_flag_summary,
        "geographic_quality_summary": geo_summary,
        "final_dtype_confirmations": confirm_selected_dtypes(df),
    }

    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(make_json_safe(report), indent=2),
        encoding="utf-8",
    )

    print_section(f"{dataset_name.upper()} RESULTS")
    print(f"Rows before cleaning: {row_count_before:,}")
    print(f"Rows after cleaning: {len(df):,}")
    print(f"Cleaned dataset saved to: {output_path}")
    print(f"Summary report saved to: {report_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean the week3 sold and listed datasets into analysis-ready CSVs and write transformation summaries."
    )
    parser.add_argument("--sold-input", type=Path, default=DEFAULT_SOLD_INPUT_PATH, help="Sold input CSV to clean.")
    parser.add_argument("--listed-input", type=Path, default=DEFAULT_LISTED_INPUT_PATH, help="Listed input CSV to clean.")
    parser.add_argument("--sold-output", type=Path, default=DEFAULT_SOLD_OUTPUT_PATH, help="Cleaned sold output CSV path.")
    parser.add_argument("--listed-output", type=Path, default=DEFAULT_LISTED_OUTPUT_PATH, help="Cleaned listed output CSV path.")
    parser.add_argument(
        "--sold-report-output",
        type=Path,
        default=DEFAULT_SOLD_REPORT_PATH,
        help="Sold JSON summary output path.",
    )
    parser.add_argument(
        "--listed-report-output",
        type=Path,
        default=DEFAULT_LISTED_REPORT_PATH,
        help="Listed JSON summary output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.sold_input, args.sold_output, args.sold_report_output, "sold")
    run_pipeline(args.listed_input, args.listed_output, args.listed_report_output, "listed")


if __name__ == "__main__":
    main()
