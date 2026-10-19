from __future__ import annotations

from pathlib import Path

import pandas as pd

from ebus.model.entities import Trip


def read_gtfs(gtfs_dir: str | Path) -> dict[str, pd.DataFrame]:
    raise NotImplementedError


def active_service_ids(gtfs: dict[str, pd.DataFrame], service_date: str) -> set[str]:
    raise NotImplementedError


def expand_frequencies(gtfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    raise NotImplementedError


def extract_trips(
    gtfs: dict[str, pd.DataFrame],
    service_date: str,
    routes: list[str] | None = None,
    time_window: tuple[int, int] | None = None,
) -> list[Trip]:
    raise NotImplementedError
