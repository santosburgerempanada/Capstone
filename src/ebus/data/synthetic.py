"""Generador de instancias sintéticas para desarrollar y probar mientras no hay datos de Santiago."""

from __future__ import annotations

import math
import random
from dataclasses import replace

from ebus.config import DEFAULT_CONFIG_PATH, load_config
from ebus.data.tariffs import tariff_from_blocks
from ebus.model.entities import BusType, Depot, Instance, Tariff, Trip

def _default_tariff() -> Tariff:
    """Tarifa de `costs` en config/default.yaml (`tou_prices` y `overnight_price_per_kwh`)."""
    costs = load_config(DEFAULT_CONFIG_PATH)["costs"]
    return tariff_from_blocks(costs["tou_prices"], overnight_price=costs["overnight_price_per_kwh"])


def _default_deadhead_cost_per_km(deadhead_cost_per_km: float | None) -> float:
    """Costo de deadhead por km: el explícito, o `costs.deadhead_cost_per_km` de config/default.yaml."""
    if deadhead_cost_per_km is not None:
        return deadhead_cost_per_km
    return load_config(DEFAULT_CONFIG_PATH)["costs"]["deadhead_cost_per_km"]


def _default_bus_type(
    consumption_kwh_per_km: float | None = None, deadhead_consumption_factor: float | None = None
) -> BusType:
    """Tipo 1 de la Tabla 6 del paper (batería y carga). El consumo sale de `battery` y el costo fijo por bus de
    `costs.vehicle_fixed_cost` en config/default.yaml, salvo que el consumo se pase explícito."""
    cfg = load_config(DEFAULT_CONFIG_PATH)
    battery_cfg = cfg["battery"]
    if consumption_kwh_per_km is None:
        consumption_kwh_per_km = battery_cfg["consumption_kwh_per_km"]
    if deadhead_consumption_factor is None:
        deadhead_consumption_factor = battery_cfg["deadhead_consumption_factor"]
    return BusType(
        id="type1", battery_kwh=200, max_charge_kw=100,
        trip_kwh_per_km=consumption_kwh_per_km,
        deadhead_kwh_per_km=consumption_kwh_per_km * deadhead_consumption_factor,
        fixed_cost_per_bus=cfg["costs"]["vehicle_fixed_cost"],
    )


def _deadhead_matrix(
    coords: dict[str, tuple[float, float]], speed_kmh: float = 20.0, detour_factor: float = 1.3
) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], float]]:
    """d_ij euclidiano: distancia en el plano × factor de desvío / velocidad. Devuelve
    (minutos, km) por par ordenado."""
    matrix: dict[tuple[str, str], int] = {}
    km_matrix: dict[tuple[str, str], float] = {}
    for i, (xi, yi) in coords.items():
        for j, (xj, yj) in coords.items():
            if i == j:
                continue
            dist_km = math.hypot(xi - xj, yi - yj) * detour_factor
            matrix[(i, j)] = max(1, round(dist_km / speed_kmh * 60))
            km_matrix[(i, j)] = dist_km
    return matrix, km_matrix


def paper_figure1_instance(
    consumption_kwh_per_km: float | None = None,
    deadhead_consumption_factor: float | None = None,
    soc_initial: float | None = None,
    deadhead_cost_per_km: float | None = None,
) -> Instance:
    """Instancia inspirada en la Fig. 1 del paper: 3 viajes, 4 estaciones, 1 depósito.

    T1 termina en S2 y T3 empieza en S2 con holgura suficiente -> conectables (igual que la
    conexión verde punteada de la figura). T2 empieza en S4; la distancia de S2 a S4 excede
    la holgura entre el fin de T1 y el inicio de T2 -> no conectables (igual que en la figura).

    Nota: no tenemos la imagen del paper, solo el texto de la leyenda, así que las coordenadas
    y horarios son una reconstrucción razonable de esa descripción, no una copia exacta de la
    figura. Sirve para probar `model/network.py` y los métodos de solución con una instancia
    mínima y legible a mano.
    """
    coords = {
        "D1": (0.0, 0.0),
        "S1": (1.0, 0.0),
        "S2": (1.0, 3.0),
        "S3": (6.0, 7.0),
        "S4": (1.0, 7.0),
    }
    deadhead, deadhead_km = _deadhead_matrix(coords)

    trips = [
        Trip(id="T1", route_id="R1", start_station="S1", end_station="S2",
             start_time=480, end_time=510, distance_km=6.0),
        Trip(id="T2", route_id="R2", start_station="S4", end_station="S3",
             start_time=515, end_time=545, distance_km=8.0),
        Trip(id="T3", route_id="R1", start_station="S2", end_station="S1",
             start_time=540, end_time=570, distance_km=6.0),
    ]
    depot = Depot(id="D1", lat=coords["D1"][1], lon=coords["D1"][0], n_chargers=2, power_kw=200)

    return Instance(
        trips=trips,
        depots=[depot],
        bus_type=_default_bus_type(consumption_kwh_per_km, deadhead_consumption_factor),
        tariff=_default_tariff(),
        deadhead_min=deadhead,
        deadhead_km=deadhead_km,
        deadhead_cost_per_km=_default_deadhead_cost_per_km(deadhead_cost_per_km),
        horizon_start=270,
        horizon_end=1410,
        soc_initial=soc_initial,
    )


def generate_instance(
    n_trips: int,
    n_stations: int,
    n_depots: int,
    horizon: tuple[int, int] = (270, 1410),
    seed: int = 0,
    consumption_kwh_per_km: float | None = None,
    deadhead_consumption_factor: float | None = None,
    soc_initial: float | None = None,
    deadhead_cost_per_km: float | None = None,
    battery_kwh: float | None = None,
    n_chargers: int | None = None,
) -> Instance:
    """Itinerario aleatorio con estaciones en un plano, d_ij euclidiano y la tarifa por defecto
    del paper. Útil para probar con más viajes que la instancia de la Fig. 1 sin depender de
    datos reales.

    `battery_kwh` y `n_chargers` permiten armar instancias que obliguen a cargar (batería chica) y
    que hagan pesar el límite de cargadores; `None` deja los valores por defecto."""
    rng = random.Random(seed)
    start, end = horizon

    stations = [f"S{i+1}" for i in range(n_stations)]
    depots = [f"D{i+1}" for i in range(n_depots)]
    coords = {name: (rng.uniform(0, 8), rng.uniform(0, 8)) for name in stations + depots}
    deadhead, deadhead_km = _deadhead_matrix(coords)
    margin = 60

    trips = []
    for k in range(n_trips):
        origin, dest = rng.sample(stations, 2)
        trip_start = rng.randint(start + margin, end - margin - 45)
        duration = rng.randint(15, 45)
        trips.append(Trip(
            id=f"T{k+1}", route_id=f"R{rng.randint(1, max(1, n_stations // 2))}",
            start_station=origin, end_station=dest,
            start_time=trip_start, end_time=trip_start + duration,
            distance_km=round(rng.uniform(3, 12), 1),
        ))

    depot_objs = [
        Depot(id=d, lat=coords[d][1], lon=coords[d][0], n_chargers=rng.randint(2, 6),
              power_kw=float(rng.randint(100, 400)))
        for d in depots
    ]
    if n_chargers is not None:
        depot_objs = [replace(d, n_chargers=n_chargers) for d in depot_objs]
    bus_type = _default_bus_type(consumption_kwh_per_km, deadhead_consumption_factor)
    if battery_kwh is not None:
        bus_type = replace(bus_type, battery_kwh=battery_kwh)

    return Instance(
        trips=trips,
        depots=depot_objs,
        bus_type=bus_type,
        tariff=_default_tariff(),
        deadhead_min=deadhead,
        deadhead_km=deadhead_km,
        deadhead_cost_per_km=_default_deadhead_cost_per_km(deadhead_cost_per_km),
        horizon_start=start,
        horizon_end=end,
        soc_initial=soc_initial,
    )
