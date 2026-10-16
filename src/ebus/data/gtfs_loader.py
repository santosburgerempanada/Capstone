"""Lectura del GTFS de DTPM y conversión a la lista de viajes del itinerario (Paso I)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ebus.model.entities import Trip


def read_gtfs(gtfs_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Lee las tablas GTFS (stops, routes, trips, stop_times, calendar, frequencies si existe)."""
    raise NotImplementedError


def active_service_ids(gtfs: dict[str, pd.DataFrame], service_date: str) -> set[str]:
    """Servicios vigentes en `service_date` según calendar / calendar_dates."""
    raise NotImplementedError


def expand_frequencies(gtfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Si el GTFS define servicios por `frequencies.txt`, genera un viaje por cada salida."""
    raise NotImplementedError


def extract_trips(
    gtfs: dict[str, pd.DataFrame],
    service_date: str,
    routes: list[str] | None = None,
    time_window: tuple[int, int] | None = None,
) -> list[Trip]:
    """Viajes con terminal de inicio/fin, horas (min desde medianoche) y distancia (km, desde shapes)."""
    raise NotImplementedError
