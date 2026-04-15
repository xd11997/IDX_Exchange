from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from week2_common import (
    drop_high_missing_columns,
    get_output_dir,
    infer_dataset_kind,
    load_csv,
    missing_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description='Week 2 EDA profile step')
    parser.add_argument('input_csv', help='Path to listed or sold CSV')
    parser.add_argument('--output-root', default='processed', help='Root output directory')
    args = parser.parse_args()

    df = load_csv(args.input_csv)
    out_dir = get_output_dir(args.input_csv, args.output_root)

    print(f'Input file: {args.input_csv}')
    print(f'Rows, columns: {df.shape}')
    print('\nDtype counts:')
    print(df.dtypes.value_counts())

    report = missing_report(df)
    report_path = out_dir / 'tables' / 'missing_report.csv'
    report.to_csv(report_path)

    high_missing = report[report['missing_flag'] == 'high_missing']
    high_missing_path = out_dir / 'tables' / 'high_missing_columns.csv'
    high_missing.to_csv(high_missing_path)

    print('\nHigh-missing columns (>90%):')
    print(high_missing[['missing_counts', 'missing_pct']])

    # Keep important analysis fields if they happen to be sparse in one dataset.
    protected_cols = [
        'ClosePrice', 'ListPrice', 'OriginalListPrice', 'DaysOnMarket',
        'CountyOrParish', 'PropertyType', 'CloseDate', 'ListingContractDate'
    ]
    cleaned_df, dropped_cols = drop_high_missing_columns(df, report, protected_cols=protected_cols)

    print(f'\nDropped {len(dropped_cols)} high-missing columns (after protection).')
    if dropped_cols:
        print(dropped_cols)

    kind = infer_dataset_kind(args.input_csv)
    final_path = Path(args.output_root) / kind / f'{kind}_cleaned_v2.csv'
    final_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(final_path, index=False)

    print(f'\nSaved missing report to: {report_path}')
    print(f'Saved cleaned dataset v2 to: {final_path}')


if __name__ == '__main__':
    main()
