
from pathlib import Path
import sys

import pandas as pd

# Make the existing src package importable from this script.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.tep_loader import audit_all_datasets, get_default_data_dir

def main():
    data_dir = get_default_data_dir()
    print(f"Dataset directory: {data_dir}")

    report = audit_all_datasets(data_dir)

    columns = [
        "file_name",
        "raw_shape",
        "documented_shape",
        "dimension_matches_doc",
        "is_transposed",
        "final_shape",
        "nan_count",
        "inf_count",
        "min_value",
        "max_value",
    ]

    print("\nDATASET AUDIT")
    print(report[columns].to_string(index=False))

    output_path = PROJECT_ROOT / "dataset_audit_report.csv"
    report.to_csv(output_path, index=False)

    print(f"\nFiles audited: {len(report)}")
    print(f"Report saved to: {output_path}")

    print("\nRaw dimensions matching documentation:")
    print(int(report["dimension_matches_doc"].sum()))

    print("\nFiles with NaN values:")
    print(int((report["nan_count"] > 0).sum()))

    print("\nFiles with infinite values:")
    print(int((report["inf_count"] > 0).sum()))

if __name__ == "__main__":
    main()