
"""Tennessee Eastman Process (TEP) Data Ingestion and Integrity Validation Module.

Author: Team B4 (23MNG336: Operational Research)
Supervised for: Chemical Manufacturing Energy and Cost Optimization
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Standard 52 TEP process column tags
MEASURED_VARS: List[str] = [f"XMEAS_{i:02d}" for i in range(1, 42)]
MANIPULATED_VARS: List[str] = [f"XMV_{i:02d}" for i in range(1, 12)]
TEP_COLUMN_NAMES: List[str] = MEASURED_VARS + MANIPULATED_VARS

# Default data directory location within the workspace
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / (
    "The Tennessee Eastman process (TEP)/"
    "The Tennessee Eastman process (TEP)/TEdata/TEdata"
)


def get_default_data_dir() -> Path:
    """Return the resolved default dataset directory."""
    if not DEFAULT_DATA_DIR.exists():
        raise FileNotFoundError(
            f"TEP benchmark data directory not found at: {DEFAULT_DATA_DIR}"
        )
    return DEFAULT_DATA_DIR


def resolve_file_path(
    dataset_id: str,
    data_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve a dataset path, accepting IDs with or without .dat."""
    base_dir = Path(data_dir) if data_dir else get_default_data_dir()
    clean_id = dataset_id.strip().lower()

    if not clean_id.endswith(".dat"):
        clean_id += ".dat"

    file_path = base_dir / clean_id

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file '{clean_id}' does not exist in {base_dir}"
        )

    return file_path


def _orient_tep_array(
    raw_array: np.ndarray,
    file_name: str,
) -> Tuple[np.ndarray, bool]:
    """Return data in samples x 52 variables orientation."""
    if raw_array.ndim != 2:
        raise ValueError(
            f"Dataset '{file_name}' must be two-dimensional; "
            f"received shape {raw_array.shape}."
        )

    raw_shape = raw_array.shape

    if raw_shape[1] == 52:
        return raw_array, False

    if raw_shape[0] == 52:
        logger.info(
            "Dataset '%s': detected transposed layout %s.",
            file_name,
            raw_shape,
        )
        return raw_array.T, True

    raise ValueError(
        f"Dataset '{file_name}' has unexpected dimensions {raw_shape}. "
        "Neither dimension matches the expected 52 variables."
    )


def load_tep_dataset(
    dataset_id: str,
    data_dir: Optional[Union[str, Path]] = None,
    verify_integrity: bool = True,
) -> pd.DataFrame:
    """Load a TEP dataset with standardized columns and integrity checks."""
    file_path = resolve_file_path(dataset_id, data_dir)
    raw_array = np.loadtxt(file_path)

    data_array, is_transposed = _orient_tep_array(
        raw_array,
        file_path.name,
    )

    if verify_integrity:
        nan_count = int(np.isnan(data_array).sum())
        inf_count = int(np.isinf(data_array).sum())

        if nan_count:
            raise ValueError(
                f"Integrity check failed: {nan_count} NaN values "
                f"in {file_path.name}"
            )

        if inf_count:
            raise ValueError(
                f"Integrity check failed: {inf_count} infinite values "
                f"in {file_path.name}"
            )

    df = pd.DataFrame(data_array, columns=TEP_COLUMN_NAMES)
    df.attrs["dataset_id"] = file_path.stem
    df.attrs["is_transposed_origin"] = is_transposed
    df.attrs["raw_shape"] = raw_array.shape
    df.attrs["is_fault"] = not file_path.stem.startswith("d00")

    return df


def audit_dataset(
    dataset_id: str,
    data_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Union[str, int, float, bool, Tuple[int, int]]]:
    """Audit dimensions and numerical integrity of a single TEP file."""
    file_path = resolve_file_path(dataset_id, data_dir)
    raw_array = np.loadtxt(file_path)
    raw_shape = raw_array.shape
    stem = file_path.stem

    data_array, is_transposed = _orient_tep_array(
        raw_array,
        file_path.name,
    )
    processed_shape = data_array.shape

    # Expected dimensions for the supplied benchmark files.
    # The supplied d00.dat is an observed 500 x 52 dataset after transpose.
    if "_te" in stem:
        expected_shape = (960, 52)
        condition_type = (
            "Normal Testing"
            if stem == "d00_te"
            else f"Fault {stem[1:3]} Testing"
        )
    elif stem == "d00":
        expected_shape = (500, 52)
        condition_type = "Normal Training"
    else:
        expected_shape = (480, 52)
        condition_type = f"Fault {stem[1:3]} Training"

    return {
        "file_name": file_path.name,
        "dataset_id": stem,
        "condition_type": condition_type,
        "is_fault": not stem.startswith("d00"),
        "raw_shape": raw_shape,
        "documented_shape": expected_shape,
        "dimension_matches_doc": processed_shape == expected_shape,
        "is_transposed": is_transposed,
        "final_shape": processed_shape,
        "nan_count": int(np.isnan(data_array).sum()),
        "inf_count": int(np.isinf(data_array).sum()),
        "min_value": float(np.min(data_array)),
        "max_value": float(np.max(data_array)),
    }


def audit_all_datasets(
    data_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Audit all available d*.dat benchmark files."""
    base_dir = Path(data_dir) if data_dir else get_default_data_dir()
    all_files = sorted(base_dir.glob("d*.dat"))

    if not all_files:
        raise FileNotFoundError(
            f"No .dat files found in {base_dir}"
        )

    audit_records = [
        audit_dataset(fp.stem, data_dir=base_dir)
        for fp in all_files
    ]

    return pd.DataFrame(audit_records)