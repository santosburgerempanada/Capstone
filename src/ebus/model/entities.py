"""Entidades del problema. Los tiempos se expresan en minutos desde medianoche y el SOC como
fracción de la batería (0-1), con soc_min = 0.1 y soc_max = 1.0 (datos del bus; el paper usa soc_min = 0.2)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np


@dataclass(frozen=True)
class Station:
    """Paradero. `id` es el stop_id del GTFS tal cual (str, p. ej. "PB1607"); nunca se
    convierte a número. Las coordenadas salen de stops.txt (stop_lat, stop_lon)."""

    id: str
    lat: float
    lon: float

    def distance_km_to(self, other: Station) -> float:
        """Distancia en línea recta (haversine) en km."""
        r = 6371.0088
        p1, p2 = math.radians(self.lat), math.radians(other.lat)
        dphi = p2 - p1
        dlmb = math.radians(other.lon - self.lon)
        a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
        return 2 * r * math.asin(math.sqrt(a))


@dataclass(frozen=True)
class Trip:
    """Viaje del itinerario (arco de A_T)."""

    id: str
    route_id: str
    start_station: str
    end_station: str
    start_time: int
    end_time: int
    distance_km: float
    shape_id: str | None = None
    terminal: str | None = None
    block_id: str | None = None

    def __post_init__(self) -> None:
        if self.end_time < self.start_time:
            raise ValueError(f"Trip {self.id}: end_time ({self.end_time}) < start_time ({self.start_time})")
        if self.distance_km < 0:
            raise ValueError(f"Trip {self.id}: distance_km negativo ({self.distance_km})")

    @property
    def duration(self) -> int:
        return self.end_time - self.start_time

    def energy_kwh(self, bus: BusType) -> float:
        """Energía del viaje = distancia × consumo por km del bus. No se guarda en el viaje:
        se calcula, así hay una sola fuente de verdad (BusType)."""
        return self.distance_km * bus.trip_kwh_per_km


@dataclass(frozen=True)
class Depot:
    id: str
    lat: float
    lon: float
    n_chargers: int
    power_kw: float
    capacity: int | None = None


@dataclass(frozen=True)
class BusType:
    id: str
    battery_kwh: float
    max_charge_kw: float
    trip_kwh_per_km: float
    deadhead_kwh_per_km: float
    fixed_cost_per_bus: float

    def usable_kwh(self, soc_min: float, soc_max: float) -> float:
        """Energía utilizable entre los límites de SOC (ej. 350 × (1.0 − 0.2) = 280 kWh)."""
        return self.battery_kwh * (soc_max - soc_min)


@dataclass
class Tariff:
    """Precio de energía por minuto del día (e_c^t) y precio nocturno (e_c^0)."""

    price_per_kwh_by_minute: np.ndarray
    overnight_price_per_kwh: float

    def price_at(self, minute: int) -> float:
        return float(self.price_per_kwh_by_minute[minute % 1440])


@dataclass
class Instance:
    trips: list[Trip]
    depots: list[Depot]
    bus_type: BusType
    tariff: Tariff
    deadhead_min: dict[tuple[str, str], int]
    deadhead_km: dict[tuple[str, str], float]
    deadhead_cost_per_km: float
    horizon_start: int
    horizon_end: int
    time_step: int = 15
    soc_min: float = 0.1
    soc_max: float = 1.0
    soc_initial: float | None = None
    station_coords: dict[str, Station] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.soc_initial is None:
            self.soc_initial = self.soc_max
        if not self.soc_min <= self.soc_initial <= self.soc_max:
            raise ValueError(
                f"soc_initial ({self.soc_initial}) debe estar entre soc_min ({self.soc_min}) "
                f"y soc_max ({self.soc_max})"
            )

    @property
    def stations(self) -> set[str]:
        return {t.start_station for t in self.trips} | {t.end_station for t in self.trips}

    def station(self, station_id: str) -> Station:
        """Paradero con coordenadas. KeyError si la instancia no las trae."""
        return self.station_coords[station_id]

    def deadhead(self, i: str, j: str) -> int:
        return 0 if i == j else self.deadhead_min[(i, j)]

    def deadhead_distance_km(self, i: str, j: str) -> float:
        return 0.0 if i == j else self.deadhead_km[(i, j)]

    def slack(self, a: Trip, b: Trip) -> int:
        """Holgura entre el fin de `a` y el inicio de `b`, en minutos (negativa si se solapan)."""
        return b.start_time - a.end_time

    def can_connect(self, a: Trip, b: Trip) -> bool:
        """Un mismo bus puede hacer `b` justo después de `a`: la holgura alcanza para el deadhead
        desde el fin de `a` hasta el inicio de `b`. Solo mira el tiempo, no la batería."""
        return self.slack(a, b) >= self.deadhead(a.end_station, b.start_station)

    def nearest_depot(self, station: str) -> Depot:
        """Depósito con menor tiempo en vacío hasta/desde `station`. Utilidad pura (no depende
        de ningún método de solución): la usan tanto el baseline greedy como, para la
        reasignación de depósitos, `heuristics/depot_assignment.py`."""
        return min(self.depots, key=lambda d: self.deadhead(d.id, station))


@dataclass
class RechargeEvent:
    depot_id: str
    start_time: int
    end_time: int
    power_kw: float
    flag: int = 0

    @property
    def duration(self) -> int:
        return self.end_time - self.start_time


@dataclass
class TripChain:
    """Plan diario de un bus: secuencia de viajes, cargas y SOC antes/después de cada viaje (Fig. 3)."""

    depot_id: str
    trips: list[Trip] = field(default_factory=list)
    recharges: dict[int, list[RechargeEvent]] = field(default_factory=dict)
    soc_before: list[float] = field(default_factory=list)
    soc_after: list[float] = field(default_factory=list)
    return_depot_id: str | None = None
    bus_id: str | None = None

    def copy(self) -> TripChain:
        """Copia independiente: listas, diccionario de cargas y cada `RechargeEvent` son nuevos, así que
        modificar la copia no toca el original. Los `Trip` se comparten porque son inmutables. Conserva el
        formato de cada entrada de `recharges` (lista de segmentos o evento suelto, formato antiguo)."""
        return TripChain(
            depot_id=self.depot_id,
            trips=list(self.trips),
            recharges={
                k: [replace(e) for e in v] if isinstance(v, list) else replace(v) for k, v in self.recharges.items()
            },
            soc_before=list(self.soc_before),
            soc_after=list(self.soc_after),
            return_depot_id=self.return_depot_id,
            bus_id=self.bus_id,
        )


@dataclass
class Solution:
    chains: list[TripChain]
    unassigned: list[Trip] = field(default_factory=list)
    cost: float | None = None

    @property
    def n_buses(self) -> int:
        return len(self.chains)

    def copy(self) -> Solution:
        """Copia independiente de la solución (equivale a `copy.deepcopy`, pero comparte los `Trip`,
        que son inmutables). Los operadores destroy la usan al empezar, porque la librería `alns` no copia."""
        return Solution(chains=[c.copy() for c in self.chains], unassigned=list(self.unassigned), cost=self.cost)


def recharge_events(chain: TripChain, k: int) -> list[RechargeEvent]:
    """Segments in stored chronological order; accepts legacy single events.

    A zero-duration, zero-power event can mark a depot visit without charging,
    preserving its deadheads when extracting a network path.
    """
    events = chain.recharges.get(k, [])
    return [events] if isinstance(events, RechargeEvent) else events


def all_recharge_events(chain: TripChain) -> list[RechargeEvent]:
    return [event for k in sorted(chain.recharges) for event in recharge_events(chain, k)]
