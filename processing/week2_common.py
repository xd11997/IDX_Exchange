from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


def infer_dataset_kind(input_path: str | Path) -> str:
    """Infer 'listed' or 'sold' from the input path."""
    path = Path(input_path)
    parts = [p.lower() for p in path.parts]
    name = path.name.lower()

    if 'sold' in parts or 'sold' in name:
        return 'sold'
    if 'listed' in parts or 'listed' in name:
        return 'listed'
    return 'listed'


def get_output_dir(input_path: str | Path, output_root: str | Path | None = None) -> Path:
    """Create and return processed/<kind>/eda_outputs."""
    kind = infer_dataset_kind(input_path)
    root = Path(output_root) if output_root is not None else Path('processed')
    out_dir = root / kind / 'eda_outputs'
    (out_dir / 'tables').mkdir(parents=True, exist_ok=True)
    (out_dir / 'plots').mkdir(parents=True, exist_ok=True)
    (out_dir / 'logs').mkdir(parents=True, exist_ok=True)
    return out_dir


def load_csv(input_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(input_path)


def existing_cols(df: pd.DataFrame, cols: Sequence[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    missing_count = df.isna().sum()
    missing_pct = missing_count / len(df) * 100 if len(df) else 0
    report = pd.DataFrame({'missing_counts': missing_count, 'missing_pct': missing_pct})
    report = report.sort_values(by='missing_pct', ascending=False)
    report['missing_flag'] = ''
    report.loc[report['missing_pct'] > 90, 'missing_flag'] = 'high_missing'
    return report


def drop_high_missing_columns(
    df: pd.DataFrame,
    report: pd.DataFrame,
    protected_cols: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    protected = set(protected_cols or [])
    cols_to_drop = [
        col for col in report.index.tolist()
        if report.loc[col, 'missing_flag'] == 'high_missing' and col not in protected
    ]
    cleaned = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    return cleaned, cols_to_drop


def safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce')
