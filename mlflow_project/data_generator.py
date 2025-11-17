"""
Utilities for creating synthetic classification data sets and saving them to disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.datasets import make_classification

DEFAULT_COLUMNS = [f"f{i}" for i in range(1, 11)]


def generate_classification_data(
    *,
    n_samples: int = 1000,
    n_features: int = 10,
    n_informative: int | None = None,
    n_redundant: int = 0,
    n_classes: int = 2,
    random_state: int = 42,
    noise: float = 0.0,
    save_path: str | Path | None = None,
) -> Tuple[pd.DataFrame, str | None]:
    """
    Generate a synthetic classification dataset and optionally persist it to CSV.

    Returns:
        tuple(DataFrame, path_str_or_none)
    """
    n_informative = n_informative or max(2, n_features // 2)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        flip_y=noise,
        random_state=random_state,
    )

    columns = DEFAULT_COLUMNS[:n_features]
    df = pd.DataFrame(X, columns=columns)
    df["target"] = y

    saved_path = None
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        saved_path = str(save_path.resolve())

    return df, saved_path
