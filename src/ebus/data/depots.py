"""Depósitos / electroterminales con su número de cargadores C_d y potencia P_d."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ebus.model.entities import Depot


def load_depots(depots_path: str | Path, charger_capacity_path: str | Path) -> list[Depot]:
    """Une `depots.csv` (ubicación y potencia `P_d`) con `charger_capacity.csv` (cargadores
    simultáneos `C_d` = `max_simultaneous_charging`). Ver docs/datos/datos_requeridos.md."""
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
    """{nombre del electroterminal: depot_id}. Los viajes de la partición traen el nombre del
    terminal ("Vespucio Norte"); los depósitos y la matriz de deadheads usan el id. Acepta la
    columna `name` (depots.csv) o `nombre` (diccionario)."""
    df = pd.read_csv(depots_path, dtype={"depot_id": str})
    columna = "name" if "name" in df.columns else "nombre"
    if df["depot_id"].duplicated().any() or df[columna].duplicated().any():
        raise ValueError(f"{depots_path}: ids o nombres de depósito repetidos")
    return dict(zip(df[columna], df["depot_id"]))


def select_depots(depots: list[Depot], mode: str) -> list[Depot]:
    """Configuraciones de la Sec. 4.3: 'single' (depósito central), 'few', 'all'."""
    raise NotImplementedError
