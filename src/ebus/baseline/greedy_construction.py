from __future__ import annotations

from ebus.data.fleet import deadhead_energy_soc, trip_energy_soc
from ebus.heuristics.depot_assignment import assign_return_depots
from ebus.model.entities import Instance, RechargeEvent, Solution, Trip, TripChain


def _overlapping(ledger: dict[str, list[RechargeEvent]], depot_id: str, ts: int, te: int) -> list[RechargeEvent]:
    return [e for e in ledger.get(depot_id, []) if e.start_time < te and e.end_time > ts]


def _available_power(events: list[RechargeEvent], ts: int, te: int, depot_power_kw: float) -> float:
    if not events:
        return depot_power_kw
    breakpoints = sorted({max(e.start_time, ts) for e in events} | {min(e.end_time, te) for e in events})
    peak = max(
        sum(e.power_kw for e in events if e.start_time <= t < e.end_time)
        for t in breakpoints[:-1]
    )
    return depot_power_kw - peak


def _return_reserve(trip: Trip, inst: Instance) -> float:
    depot = inst.nearest_depot(trip.end_station)
    return deadhead_energy_soc(inst.deadhead_distance_km(trip.end_station, depot.id), inst.bus_type)


def _try_append(
    chain: TripChain, trip: Trip, inst: Instance, ledger: dict[str, list[RechargeEvent]], reserve: float
) -> tuple[float, float, float, RechargeEvent | None] | None:
    last = chain.trips[-1]
    gap = trip.start_time - last.end_time
    bus = inst.bus_type
    soc_before_chain = chain.soc_after[-1]

    direct_deadhead = inst.deadhead(last.end_station, trip.start_station)
    if gap >= direct_deadhead:
        soc_before_trip = soc_before_chain - deadhead_energy_soc(inst.deadhead_distance_km(last.end_station, trip.start_station), bus)
        soc_after_trip = soc_before_trip - trip_energy_soc(trip, bus)
        if soc_before_trip >= inst.soc_min - 1e-9 and soc_after_trip - reserve >= inst.soc_min - 1e-9:
            cost = inst.deadhead_distance_km(last.end_station, trip.start_station) * inst.deadhead_cost_per_km
            return cost, soc_before_trip, soc_after_trip, None

    depot = inst.nearest_depot(last.end_station)
    to_depot = inst.deadhead(last.end_station, depot.id)
    from_depot = inst.deadhead(depot.id, trip.start_station)
    charge_window = gap - to_depot - from_depot
    if charge_window <= 0:
        return None

    soc_at_depot = soc_before_chain - deadhead_energy_soc(inst.deadhead_distance_km(last.end_station, depot.id), bus)
    if soc_at_depot < inst.soc_min - 1e-9:
        return None

    ts, te = last.end_time + to_depot, last.end_time + to_depot + charge_window
    others = _overlapping(ledger, depot.id, ts, te)
    if len(others) + 1 > depot.n_chargers:
        return None

    available_power = _available_power(others, ts, te, depot.power_kw)
    if available_power <= 1e-9:
        return None

    power = min(bus.max_charge_kw, available_power)
    charged = power * (charge_window / 60) / bus.battery_kwh
    soc_after_charge = min(inst.soc_max, soc_at_depot + charged)

    soc_before_trip = soc_after_charge - deadhead_energy_soc(inst.deadhead_distance_km(depot.id, trip.start_station), bus)
    soc_after_trip = soc_before_trip - trip_energy_soc(trip, bus)
    if soc_before_trip < inst.soc_min - 1e-9 or soc_after_trip - reserve < inst.soc_min - 1e-9:
        return None

    recharge = RechargeEvent(depot_id=depot.id, start_time=ts, end_time=te, power_kw=power)
    cost = (
        inst.deadhead_distance_km(last.end_station, depot.id)
        + inst.deadhead_distance_km(depot.id, trip.start_station)
    ) * inst.deadhead_cost_per_km
    return cost, soc_before_trip, soc_after_trip, recharge


def _commit_append(
    chain: TripChain, trip: Trip, soc_before_trip: float, soc_after_trip: float,
    recharge: RechargeEvent | None, ledger: dict[str, list[RechargeEvent]],
) -> None:
    if recharge is not None:
        chain.recharges[len(chain.trips)] = recharge
        ledger.setdefault(recharge.depot_id, []).append(recharge)
    chain.trips.append(trip)
    chain.soc_before.append(soc_before_trip)
    chain.soc_after.append(soc_after_trip)


def _open_chain(trip: Trip, inst: Instance) -> TripChain:
    depot = inst.nearest_depot(trip.start_station)
    to_trip = inst.deadhead(depot.id, trip.start_station)
    bus = inst.bus_type

    soc_before_trip = inst.soc_initial - deadhead_energy_soc(inst.deadhead_distance_km(depot.id, trip.start_station), bus)
    soc_after_trip = soc_before_trip - trip_energy_soc(trip, bus)

    return TripChain(
        depot_id=depot.id, trips=[trip],
        soc_before=[soc_before_trip], soc_after=[soc_after_trip],
    )


def build_baseline_solution(inst: Instance) -> Solution:
    trips_sorted = sorted(inst.trips, key=lambda t: (t.start_time, t.id))
    chains: list[TripChain] = []
    ledger: dict[str, list[RechargeEvent]] = {}

    for trip in trips_sorted:
        best: tuple[float, TripChain, float, float, RechargeEvent | None] | None = None
        reserve = _return_reserve(trip, inst)
        for chain in chains:
            result = _try_append(chain, trip, inst, ledger, reserve)
            if result is None:
                continue
            cost, soc_before_trip, soc_after_trip, recharge = result
            if best is None or cost < best[0]:
                best = (cost, chain, soc_before_trip, soc_after_trip, recharge)

        if best is not None:
            _, chain, soc_before_trip, soc_after_trip, recharge = best
            _commit_append(chain, trip, soc_before_trip, soc_after_trip, recharge, ledger)
        else:
            chain = _open_chain(trip, inst)
            chain.bus_id = f"B{len(chains) + 1:03d}"
            chains.append(chain)

    solution = Solution(chains=chains)
    assign_return_depots(solution, inst)
    return solution
