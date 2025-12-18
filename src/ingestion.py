from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import get_logger


logger = get_logger(__name__)


def load_raw_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load the raw restaurant reviews CSV reliably and standardize column names.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    logger.info("Loaded raw CSV: %s | shape=%s", csv_path.name, df.shape)
    return df