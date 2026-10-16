"""Parámetros globales del caso (data/raw/other/parameters.csv, ver parametros_descripcion.csv)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_parameters(path: str | Path) -> dict[str, float]:
    """Lee `parameter, value` y devuelve {nombre: valor}. Falla si hay nombres repetidos."""
    df = pd.read_csv(path)
    if df["parameter"].duplicated().any():
        raise ValueError(f"{path}: parámetros repetidos: {sorted(df['parameter'][df['parameter'].duplicated()])}")
    return {str(k): float(v) for k, v in zip(df["parameter"], df["value"])}
