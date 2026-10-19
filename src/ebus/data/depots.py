from __future__ import annotations

from pathlib import Path

import pandas as pd

from ebus.model.entities import Depot


def load_depots(depots_path: str | Path, charger_capacity_path: str | Path) -> list[Depot]:
    depots = pd.read_csv(depots_path, dtype={"depot_id": str})
    if depots["depot_id"].duplicated().any():
        raise ValueError(f"{depots_path}: depot_id repetidos: {sorted(depots['depot_id'][depots['depot_id'].duplicated()])}")
    chargers = pd.read_csv(charger_capacity_path, dtype={"depot_id": str}).set_index("depot_id")
    missing = sorted(set(depots["depot_id"]) - set(chargers.index))
    if missing:
        raise ValueError(f"{charger_capacity_path}: faltan los depósitos {missing}")
    return [
        Depot(
            id=row.depot_id, lat=float(row.lat), lon=float(row.lon),
            n_chargers=int(chargers.loc[row.depot_id, "max_simultaneous_charging"]),
            power_kw=float(row.power_kw),
            capacity=int(row.capacity) if "capacity" in depots.columns and pd.notna(row.capacity) else None,
        )
        for row in depots.itertuples()
    ]


def depot_ids_por_nombre(depots_path: str | Path) -> dict[str, str]:
    df = pd.read_csv(depots_path, dtype={"depot_id": str})
    columna = "name" if "name" in df.columns else "nombre"
    if df["depot_id"].duplicated().any() or df[columna].duplicated().any():
        raise ValueError(f"{depots_path}: ids o nombres de depósito repetidos")
    return dict(zip(df[columna], df["depot_id"]))


def select_depots(depots: list[Depot], mode: str) -> list[Depot]:
    raise NotImplementedError
