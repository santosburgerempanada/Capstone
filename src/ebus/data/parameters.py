from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_parameters(path: str | Path) -> dict[str, float]:
    df = pd.read_csv(path)
    if df["parameter"].duplicated().any():
        raise ValueError(f"{path}: parámetros repetidos: {sorted(df['parameter'][df['parameter'].duplicated()])}")
    return {str(k): float(v) for k, v in zip(df["parameter"], df["value"])}
