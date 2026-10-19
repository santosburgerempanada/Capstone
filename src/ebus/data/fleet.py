from __future__ import annotations

from pathlib import Path

import pandas as pd

from ebus.model.entities import BusType, Trip


def daily_depreciation(price: float, lifespan_years: float, days_per_year: int = 365) -> float:
    return price / (lifespan_years * days_per_year)


def load_bus_types(
    path: str | Path,
    consumption_kwh_per_km: float,
    deadhead_consumption_factor: float,
    fixed_cost_per_bus: float,
) -> list[BusType]:
    df = pd.read_csv(path, dtype={"type_id": str})
    return [
        BusType(
            id=row.type_id, battery_kwh=float(row.battery_kwh), max_charge_kw=float(row.charge_power_kw),
            trip_kwh_per_km=consumption_kwh_per_km,
            deadhead_kwh_per_km=consumption_kwh_per_km * deadhead_consumption_factor,
            fixed_cost_per_bus=fixed_cost_per_bus,
        )
        for row in df.itertuples()
    ]


def trip_energy_soc(trip: Trip, bus: BusType) -> float:
    return trip.energy_kwh(bus) / bus.battery_kwh


def deadhead_energy_soc(km: float, bus: BusType) -> float:
    return km * bus.deadhead_kwh_per_km / bus.battery_kwh
