from __future__ import annotations

import argparse
import io
import subprocess
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
FRED_FALLBACK_URL = "https://fred.stlouisfed.org/series/MORTGAGE30US/downloaddata/MORTGAGE30US.csv"
REQUEST_TIMEOUT_SECONDS = 5
DEFAULT_SOLD_PATH = PROJECT_ROOT / "processed/sold/sold_cleaned_v2.csv"
DEFAULT_LISTED_PATH = PROJECT_ROOT / "processed/listed/listed_cleaned_v2.csv"


def _read_url_text(url: str) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/csv,text/plain,*/*",
            "Connection": "close",
        },
    )
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return response.read().decode("utf-8", errors="replace")


def _read_url_text_with_curl(url: str) -> str:
    result = subprocess.run(
        [
            "curl",
            "-L",
            "--http1.1",
            "--max-time",
            str(REQUEST_TIMEOUT_SECONDS),
            "--fail",
            "--silent",
            "--show-error",
            "-A",
            "Mozilla/5.0",
            url,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _parse_mortgage_csv(raw_text: str) -> pd.DataFrame:
    if "<html" in raw_text.lower():
        snippet = raw_text[:200].replace("\n", " ")
        raise ValueError(
            "FRED did not return a CSV file. "
            f"The response looks like HTML instead: {snippet}"
        )

    mortgage = pd.read_csv(io.StringIO(raw_text))
    mortgage.columns = [str(col).strip() for col in mortgage.columns]

    date_col = next(
        (col for col in mortgage.columns if col.lower() in {"date", "observation_date"}),
        None,
    )
    rate_col = next((col for col in mortgage.columns if col != date_col), None)

    if date_col is None or rate_col is None:
        raise ValueError(
            "Unexpected FRED columns returned: "
            f"{list(mortgage.columns)}"
        )

    mortgage = mortgage.rename(
        columns={date_col: "date", rate_col: "rate_30yr_fixed"}
    )
    mortgage["date"] = pd.to_datetime(mortgage["date"], errors="coerce")
    mortgage["rate_30yr_fixed"] = pd.to_numeric(
        mortgage["rate_30yr_fixed"], errors="coerce"
    )
    mortgage = mortgage.dropna(subset=["date", "rate_30yr_fixed"])
    return mortgage


def fetch_mortgage_data(url: str = FRED_URL, local_csv: Path | None = None) -> pd.DataFrame:
    """Fetch weekly mortgage data directly from FRED."""
    if local_csv is not None:
        print(f"Reading mortgage data from local CSV: {local_csv}")
        raw_text = local_csv.read_text(encoding="utf-8", errors="replace")
        return _parse_mortgage_csv(raw_text)

    errors: list[str] = []

    for candidate_url in (url, FRED_FALLBACK_URL):
        for loader_name, loader in (
            ("urllib", _read_url_text),
            ("curl", _read_url_text_with_curl),
        ):
            try:
                print(f"Trying mortgage download via {loader_name}: {candidate_url}")
                raw_text = loader(candidate_url)
                print(f"Mortgage download succeeded via {loader_name}: {candidate_url}")
                return _parse_mortgage_csv(raw_text)
            except Exception as exc:
                print(f"Mortgage download failed via {loader_name}: {candidate_url}")
                errors.append(f"{loader_name} failed for {candidate_url}: {exc}")

    raise RuntimeError(
        "Unable to fetch mortgage data from FRED.\n"
        "Tip: download MORTGAGE30US.csv in your browser and rerun with:\n"
        "python3 week3.py --mortgage-input /path/to/MORTGAGE30US.csv\n"
        + "\n".join(errors)
    )


def resample_mortgage_monthly(mortgage: pd.DataFrame) -> pd.DataFrame:
    """
    Convert weekly mortgage observations into monthly averages.

    FRED publishes MORTGAGE30US weekly, so we group all weekly observations
    that fall in the same calendar month and take the mean rate for that month.
    """
    monthly = (
        mortgage.set_index("date")
        .resample("MS")
        .mean(numeric_only=True)
        .reset_index()
    )
    monthly["year_month"] = monthly["date"].dt.to_period("M").astype(str)
    return monthly[["date", "year_month", "rate_30yr_fixed"]]


def add_year_month(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Create a YYYY-MM merge key from a date column."""
    if date_col not in df.columns:
        raise KeyError(f"Column '{date_col}' not found. Available columns: {list(df.columns)}")

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out["year_month"] = out[date_col].dt.to_period("M").astype("string")
    out.loc[out[date_col].isna(), "year_month"] = pd.NA
    return out


def merge_mortgage_rates(dataset: pd.DataFrame, mortgage_monthly: pd.DataFrame) -> pd.DataFrame:
    """Left-join monthly mortgage rates onto a dataset using year_month."""
    merged = dataset.merge(
        mortgage_monthly[["year_month", "rate_30yr_fixed"]],
        on="year_month",
        how="left",
    )
    return merged


def process_dataset(
    input_path: Path,
    output_path: Path,
    date_col: str,
    mortgage_monthly: pd.DataFrame,
) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)
    df = add_year_month(df, date_col=date_col)
    merged = merge_mortgage_rates(df, mortgage_monthly)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged


def validate_no_null_rates(dataset: pd.DataFrame, dataset_name: str) -> None:
    """Validate that merged mortgage rates do not contain null values."""
    null_count = int(dataset["rate_30yr_fixed"].isna().sum())
    total_rows = len(dataset)

    print(f"Validation result for {dataset_name}: {null_count} null rate values out of {total_rows:,} rows")

    if null_count > 0:
        missing_months = (
            dataset.loc[dataset["rate_30yr_fixed"].isna(), "year_month"]
            .dropna()
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )
        raise ValueError(
            f"Validation failed for {dataset_name}: found {null_count} rows with null mortgage rates. "
            f"Missing year_month values: {missing_months}"
        )

    print(f"Validation passed for {dataset_name}: no null mortgage rates found")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch weekly mortgage rates from FRED, resample to monthly, and merge onto sold/listed datasets."
    )
    parser.add_argument("--sold-input", type=Path, default=DEFAULT_SOLD_PATH)
    parser.add_argument("--listed-input", type=Path, default=DEFAULT_LISTED_PATH)
    parser.add_argument(
        "--sold-date-col",
        default="CloseDate",
        help="Date column used to build year_month for the sold dataset.",
    )
    parser.add_argument(
        "--listed-date-col",
        default="ListingContractDate",
        help="Date column used to build year_month for the listed dataset.",
    )
    parser.add_argument(
        "--mortgage-output",
        type=Path,
        default=PROJECT_ROOT / "processed/mortgage/mortgage_monthly.csv",
        help="Where to save the monthly mortgage table.",
    )
    parser.add_argument(
        "--mortgage-input",
        type=Path,
        default=None,
        help="Optional local mortgage CSV path. If provided, skip the live FRED download.",
    )
    parser.add_argument(
        "--sold-output",
        type=Path,
        default=PROJECT_ROOT / "processed/sold/merged_sold_with_rate.csv",
    )
    parser.add_argument(
        "--listed-output",
        type=Path,
        default=PROJECT_ROOT / "processed/listed/merged_listed_with_rate.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Step 1/4: Fetching weekly mortgage data")
    mortgage_weekly = fetch_mortgage_data(local_csv=args.mortgage_input)
    print("Step 2/4: Resampling weekly mortgage data to monthly averages")
    mortgage_monthly = resample_mortgage_monthly(mortgage_weekly)

    args.mortgage_output.parent.mkdir(parents=True, exist_ok=True)
    mortgage_monthly.to_csv(args.mortgage_output, index=False)
    print(f"Saved monthly mortgage data to: {args.mortgage_output}")

    print("Step 3/4: Merging monthly mortgage rates onto sold data")
    sold_merged = process_dataset(
        input_path=args.sold_input,
        output_path=args.sold_output,
        date_col=args.sold_date_col,
        mortgage_monthly=mortgage_monthly,
    )
    print("Step 4/4: Merging monthly mortgage rates onto listed data")
    listed_merged = process_dataset(
        input_path=args.listed_input,
        output_path=args.listed_output,
        date_col=args.listed_date_col,
        mortgage_monthly=mortgage_monthly,
    )
    print("Step 5/5: Validating that merged datasets contain no null mortgage rates")
    validate_no_null_rates(sold_merged, "sold")
    validate_no_null_rates(listed_merged, "listed")

    print(f"Saved monthly mortgage data to: {args.mortgage_output}")
    print(f"Sold rows merged: {len(sold_merged):,}")
    print(f"Saved sold output to: {args.sold_output}")
    print(f"Listed rows merged: {len(listed_merged):,}")
    print(f"Saved listed output to: {args.listed_output}")


if __name__ == "__main__":
    main()
