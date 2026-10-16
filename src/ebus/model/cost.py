"""Evaluación del objetivo (1) sobre una solución en forma de cadenas de viajes (usada por
baseline, MILP y ALNS por igual: no depende de qué método generó la solución)."""

from __future__ import annotations

from dataclasses import dataclass

from ebus.model.entities import Instance, Solution, TripChain, recharge_events, all_recharge_events


@dataclass
class CostBreakdown:
    bus_fixed_cost: float
    deadhead: float
    daytime_charging: float
    overnight_charging: float

    @property
    def total(self) -> float:
        return self.bus_fixed_cost + self.deadhead + self.daytime_charging + self.overnight_charging


def chain_deadhead_km(chain: TripChain, inst: Instance) -> float:
    """Km de deadhead: salida de depósito, entre viajes (y hacia/desde un depósito si hay una
    carga intermedia, ec. 20), y retorno. No incluye los propios arcos de viaje."""
    total = 0.0
    location = chain.depot_id
    for k, trip in enumerate(chain.trips):
        for recharge in recharge_events(chain, k):
            total += inst.deadhead_distance_km(location, recharge.depot_id)
            location = recharge.depot_id
        total += inst.deadhead_distance_km(location, trip.start_station)
        location = trip.end_station

    for recharge in recharge_events(chain, len(chain.trips)):
        total += inst.deadhead_distance_km(location, recharge.depot_id)
        location = recharge.depot_id
    if chain.return_depot_id is not None:
        total += inst.deadhead_distance_km(location, chain.return_depot_id)
    return total


def soc_at_return(chain: TripChain, inst: Instance) -> float:
    """SOC con que el bus llega al depósito de retorno al final del día.

    Parte del SOC tras el último viaje y descuenta la energía del deadhead de vuelta (km × consumo en
    deadhead). Si hay una carga final (`recharges[len(trips)]`), descuenta la ida al depósito de carga,
    suma lo cargado (sin pasar de `soc_max`) y luego descuenta el tramo de ese depósito al de retorno.
    Si la cadena aún no tiene depósito de retorno asignado, no descuenta ese tramo. Se calcula a pedido
    para que nunca quede desfasado de los viajes de la cadena."""
    bus = inst.bus_type
    soc = chain.soc_after[-1]
    ubicacion = chain.trips[-1].end_station

    for carga_final in recharge_events(chain, len(chain.trips)):
        soc -= inst.deadhead_distance_km(ubicacion, carga_final.depot_id) * bus.deadhead_kwh_per_km / bus.battery_kwh
        soc = min(inst.soc_max, soc + carga_final.power_kw * carga_final.duration / 60 / bus.battery_kwh)
        ubicacion = carga_final.depot_id
    if chain.return_depot_id is not None:
        soc -= inst.deadhead_distance_km(ubicacion, chain.return_depot_id) * bus.deadhead_kwh_per_km / bus.battery_kwh
    return soc


def chain_charging_cost(chain: TripChain, inst: Instance) -> float:
    """Σ e_c^t · potencia · Δt/60 sobre cada evento de carga diurna, integrado minuto a minuto
    (igual que la suma discreta de las ecuaciones (1)/(8), sin asumir un precio único por evento)."""
    total = 0.0
    for recharge in all_recharge_events(chain):
        for minute in range(recharge.start_time, recharge.end_time):
            total += inst.tariff.price_at(minute) * recharge.power_kw / 60
    return total


def evaluate(solution: Solution, inst: Instance) -> CostBreakdown:
    """(1): costo fijo por bus + deadheads (por km) + carga diurna + carga nocturna.
    La carga nocturna cubre el déficit de SOC con el que cada bus llega al depósito al terminar su día
    (e_c^0 · (soc_max − s_{d,T-1})), tal como lo asume el paper: capacidad nocturna ilimitada. Ese SOC
    ya descuenta la energía del deadhead de retorno (`soc_at_return`).
    Recarga hasta soc_max aunque el bus haya partido con `soc_initial` < soc_max (decisión del
    equipo, ver docs/decisiones/modelo.md)."""
    bus_fixed_cost = len(solution.chains) * inst.bus_type.fixed_cost_per_bus

    deadhead_km = sum(chain_deadhead_km(c, inst) for c in solution.chains)
    deadhead_cost = deadhead_km * inst.deadhead_cost_per_km

    daytime_charging = sum(chain_charging_cost(c, inst) for c in solution.chains)

    overnight_charging = sum(
        (inst.soc_max - soc_at_return(c, inst)) * inst.bus_type.battery_kwh * inst.tariff.overnight_price_per_kwh
        for c in solution.chains
        if c.soc_after
    )

    return CostBreakdown(bus_fixed_cost, deadhead_cost, daytime_charging, overnight_charging)


def chain_variable_cost(chain: TripChain, inst: Instance) -> float:
    """Costo de una cadena sin el costo fijo del bus: deadhead (por km), carga diurna y carga nocturna hasta
    `soc_max`. Sumado sobre las cadenas, más `n_buses × costo fijo`, da `evaluate(...).total`. El repair lo usa para
    medir cuánto cuesta insertar un viaje en un bus existente (ver docs/decisiones/alns.md)."""
    deadhead = chain_deadhead_km(chain, inst) * inst.deadhead_cost_per_km
    diurna = chain_charging_cost(chain, inst)
    nocturna = (
        (inst.soc_max - soc_at_return(chain, inst)) * inst.bus_type.battery_kwh * inst.tariff.overnight_price_per_kwh
        if chain.soc_after else 0.0
    )
    return deadhead + diurna + nocturna
