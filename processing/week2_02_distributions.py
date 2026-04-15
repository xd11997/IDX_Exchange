from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from week2_common import (
    drop_high_missing_columns,
    existing_cols,
    get_output_dir,
    load_csv,
    missing_report,
    safe_to_numeric,
)


FOCUS_NUMERIC_COLS = [
    'ClosePrice', 'ListPrice', 'OriginalListPrice', 'LivingArea',
    'LotSizeAcres', 'BedroomsTotal', 'BathroomsTotalInteger', 'DaysOnMarket', 'YearBuilt'
]


def main() -> None:
    parser = argparse.ArgumentParser(description='Week 2 EDA distribution step')
    parser.add_argument('input_csv', help='Path to listed or sold CSV')
    parser.add_argument('--output-root', default='processed', help='Root output directory')
    args = parser.parse_args()

    df = load_csv(args.input_csv)
    out_dir = get_output_dir(args.input_csv, args.output_root)

    report = missing_report(df)
    cleaned_df, _ = drop_high_missing_columns(
        df,
        report,
        protected_cols=['ClosePrice', 'ListPrice', 'OriginalListPrice', 'DaysOnMarket', 'CountyOrParish', 'PropertyType'],
    )

    focus_cols = existing_cols(cleaned_df, FOCUS_NUMERIC_COLS)
    for col in focus_cols:
        cleaned_df[col] = safe_to_numeric(cleaned_df[col])

    numeric_cols_all = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
    (out_dir / 'tables' / 'numeric_columns_all.txt').write_text('\n'.join(numeric_cols_all) + ('\n' if numeric_cols_all else ''))

    # Focus histograms / boxplots, matching the notebook's idea but only for the columns that exist.
    if focus_cols:
        fig, axes = plt.subplots(len(focus_cols), 2, figsize=(16, 4 * len(focus_cols)))
        if len(focus_cols) == 1:
            axes = np.array([axes])

        for idx, col in enumerate(focus_cols):
            series = cleaned_df[col].dropna()
            axes[idx, 0].hist(series, bins=50, edgecolor='black', alpha=0.7)
            axes[idx, 0].set_title(f'Histogram: {col}')
            axes[idx, 0].set_xlabel('Value')
            axes[idx, 0].set_ylabel('Frequency')

            axes[idx, 1].boxplot(series, vert=True)
            axes[idx, 1].set_title(f'Boxplot: {col}')
            axes[idx, 1].set_ylabel('Value')

        plt.tight_layout()
        fig_path = out_dir / 'plots' / 'numeric_hist_box_focus.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved focus histogram/boxplot grid to: {fig_path}')

    # Percentile / outlier summary for all numeric columns.
    outliers_summary = []
    percentiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]

    for col in numeric_cols_all:
        valid_data = cleaned_df[col].dropna()
        if valid_data.empty:
            continue

        q1 = valid_data.quantile(0.25)
        q3 = valid_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = valid_data[(valid_data < lower_bound) | (valid_data > upper_bound)]

        extreme_lower = q1 - 3 * iqr
        extreme_upper = q3 + 3 * iqr
        extreme_outliers = valid_data[(valid_data < extreme_lower) | (valid_data > extreme_upper)]

        outliers_summary.append({
            'column': col,
            'count': int(valid_data.shape[0]),
            'mean': float(valid_data.mean()),
            'median': float(valid_data.median()),
            'std': float(valid_data.std()),
            'min': float(valid_data.min()),
            'p95': float(np.percentile(valid_data, 95)),
            'p99': float(np.percentile(valid_data, 99)),
            'max': float(valid_data.max()),
            'iqr': float(iqr),
            'outlier_count': int(outliers.shape[0]),
            'outlier_pct': float(outliers.shape[0] / valid_data.shape[0] * 100),
            'extreme_outlier_count': int(extreme_outliers.shape[0]),
            'extreme_outlier_pct': float(extreme_outliers.shape[0] / valid_data.shape[0] * 100),
        })

    outliers_df = pd.DataFrame(outliers_summary)
    outliers_path = out_dir / 'tables' / 'outlier_summary.csv'
    outliers_df.to_csv(outliers_path, index=False)
    print(f'Saved outlier summary to: {outliers_path}')

    # Save a compact percentile summary for the focus columns.
    focus_summary_rows = []
    for col in focus_cols:
        valid_data = cleaned_df[col].dropna()
        if valid_data.empty:
            continue
        row = {'column': col, 'count': int(valid_data.shape[0]), 'mean': float(valid_data.mean()), 'median': float(valid_data.median())}
        for p in percentiles:
            row[f'p{p}'] = float(np.percentile(valid_data, p))
        focus_summary_rows.append(row)

    focus_summary = pd.DataFrame(focus_summary_rows)
    focus_summary_path = out_dir / 'tables' / 'focus_numeric_summary.csv'
    focus_summary.to_csv(focus_summary_path, index=False)
    print(f'Saved focus numeric summary to: {focus_summary_path}')


if __name__ == '__main__':
    main()
