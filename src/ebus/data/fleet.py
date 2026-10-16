"""Tipos de bus eléctrico (análogo a la Tabla 6 del paper)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ebus.model.entities import BusType, Trip


def daily_depreciation(price: float, lifespan_years: float, days_per_year: int = 365) -> float:
    """e_f: costo de adquisición amortizado linealmente (el paper usa 8 años)."""
    return price / (lifespan_years * days_per_year)


def load_bus_types(
    path: str | Path,
    consumption_kwh_per_km: float,
    deadhead_consumption_factor: float,
    fixed_cost_per_bus: float,
) -> list[BusType]:
    """Lee `bus_types.csv` (batería y potencia de carga). El consumo, el factor de deadhead y el
    costo fijo por bus no vienen en ese archivo: se pasan desde la config y `parameters.csv`
    (`vehicle_fixed_cost`, que hace el papel de `e_f`; ver docs/decisiones/parametros.md)."""
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
    """Fracción de batería consumida en un viaje: distancia (km) × consumo (kWh/km) / batería (kWh).
    Se aparta del paper, que usa potencia constante por tiempo (ec. 11); ver docs/decisiones/datos.md."""
    return trip.energy_kwh(bus) / bus.battery_kwh


def deadhead_energy_soc(km: float, bus: BusType) -> float:
    """Fracción de batería consumida en un deadhead: distancia (km) × consumo (kWh/km) / batería (kWh)."""
    return km * bus.deadhead_kwh_per_km / bus.battery_kwh
