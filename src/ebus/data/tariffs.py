"""Tarifa eléctrica horaria e_c^t y nocturna e_c^0."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ebus.model.entities import Tariff


def _to_minute(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def tariff_from_blocks(blocks: list[dict], overnight_price: float) -> Tariff:
    """Construye el precio por minuto a partir de bloques {start, end, price}; admite bloques que cruzan medianoche."""
    prices = np.full(1440, np.nan)
    for b in blocks:
        start, end = _to_minute(b["start"]), _to_minute(b["end"])
        minutes = np.arange(start, end if end > start else end + 1440) % 1440
        prices[minutes] = b["price"]
    if np.isnan(prices).any():
        raise ValueError("Los bloques tarifarios no cubren las 24 horas")
    return Tariff(prices, overnight_price)


def load_santiago_tariff(path: str, overnight_price_per_kwh: float) -> Tariff:
    """Lee `electricity_prices.csv` (`period_id, start_hour, end_hour, price_usd_kwh`) y arma el
    precio por minuto. El precio nocturno `e_c^0` no viene en ese archivo y se pasa explícito
    (sin valor por defecto, para no usar uno equivocado)."""
    if overnight_price_per_kwh is None:
        raise ValueError("Falta el precio nocturno (costs.overnight_price_per_kwh): no tiene valor por defecto")
    df = pd.read_csv(path)
    blocks = [
        {"start": f"{int(r.start_hour):02d}:00", "end": f"{int(r.end_hour):02d}:00", "price": float(r.price_usd_kwh)}
        for r in df.itertuples()
    ]
    return tariff_from_blocks(blocks, overnight_price_per_kwh)
