from __future__ import annotations

from collections import Counter

from ebus.model.entities import Instance, Solution, recharge_events, all_recharge_events

_EPS = 1e-6


def _etiqueta(idx: int, chain) -> str:
    return f"Cadena {idx} (bus {chain.bus_id})" if chain.bus_id else f"Cadena {idx}"


def check_solution(sol: Solution, inst: Instance) -> list[str]:
    violations: list[str] = []
    violations += _check_coverage(sol, inst)
    violations += _check_bus_ids(sol)
    violations += _check_chain_continuity(sol, inst)
    violations += _check_soc_bounds(sol, inst)
    violations += _check_initial_soc(sol, inst)
    violations += _check_return_soc(sol, inst)
    violations += _check_soc_consistency(sol, inst)
    violations += _check_recharge_positions(sol)
    violations += _check_charging_resources(sol, inst)
    violations += _check_depot_capacity(sol, inst)
    violations += _check_depot_balance(sol, inst)
    return violations


def _check_bus_ids(sol: Solution) -> list[str]:
    counts = Counter(c.bus_id for c in sol.chains if c.bus_id)
    return [f"bus_id {bus_id} repetido en {n} cadenas" for bus_id, n in counts.items() if n > 1]


def _check_coverage(sol: Solution, inst: Instance) -> list[str]:
    violations = []
    expected = {t.id for t in inst.trips}
    counts = Counter(t.id for chain in sol.chains for t in chain.trips)

    for tid in expected:
        c = counts.get(tid, 0)
        if c == 0:
            violations.append(f"Viaje {tid} no fue asignado a ningún bus")
        elif c > 1:
            violations.append(f"Viaje {tid} fue asignado {c} veces")

    for tid in set(counts) - expected:
        violations.append(f"Viaje {tid} no pertenece a la instancia")

    if sol.unassigned:
        ids = sorted(t.id for t in sol.unassigned)
        violations.append(f"{len(ids)} viaje(s) quedaron sin asignar: {ids}")
    return violations


def _check_chain_continuity(sol: Solution, inst: Instance) -> list[str]:
    violations = []
    for idx, chain in enumerate(sol.chains):
        label = f"{_etiqueta(idx, chain)} (depósito {chain.depot_id})"
        if not chain.trips:
            violations.append(f"{label} no tiene viajes")
            continue

        location = chain.depot_id
        time = inst.horizon_start
        for k, trip in enumerate(chain.trips):
            for recharge in recharge_events(chain, k):
                to_depot = inst.deadhead(location, recharge.depot_id)
                if recharge.start_time < time + to_depot:
                    violations.append(
                        f"{label}: no alcanza el tiempo para llegar a cargar en "
                        f"{recharge.depot_id} antes de {trip.id}"
                    )
                from_depot = inst.deadhead(recharge.depot_id, trip.start_station)
                if trip.start_time < recharge.end_time + from_depot:
                    violations.append(
                        f"{label}: la carga en {recharge.depot_id} no deja tiempo para llegar a {trip.id}"
                    )
                location, time = recharge.depot_id, recharge.end_time

            deadhead = inst.deadhead(location, trip.start_station)
            if trip.start_time < time + deadhead:
                violations.append(
                    f"{label}: holgura insuficiente para llegar a {trip.id} "
                    f"(necesita {deadhead} min, hay {trip.start_time - time})"
                )
            location, time = trip.end_station, trip.end_time

        for recharge in recharge_events(chain, len(chain.trips)):
            to_depot = inst.deadhead(location, recharge.depot_id)
            if recharge.start_time < time + to_depot:
                violations.append(
                    f"{label}: no alcanza el tiempo para llegar a cargar en "
                    f"{recharge.depot_id} tras el último viaje"
                )
            location, time = recharge.depot_id, recharge.end_time

        if chain.return_depot_id is None:
            violations.append(f"{label} no tiene depósito de retorno asignado")
        else:
            deadhead = inst.deadhead(location, chain.return_depot_id)
            if time + deadhead > inst.horizon_end:
                violations.append(f"{label}: el retorno a {chain.return_depot_id} excede el horizonte operativo")
    return violations


def _check_soc_bounds(sol: Solution, inst: Instance) -> list[str]:
    violations = []
    for idx, chain in enumerate(sol.chains):
        label = _etiqueta(idx, chain)
        if len(chain.soc_before) != len(chain.trips) or len(chain.soc_after) != len(chain.trips):
            violations.append(f"{label}: SOC no registrado para todos los viajes")
            continue
        for k, trip in enumerate(chain.trips):
            for momento, soc in (("antes de", chain.soc_before[k]), ("después de", chain.soc_after[k])):
                if soc < inst.soc_min - _EPS:
                    violations.append(f"{label}: SOC {momento} {trip.id} = {soc:.3f} < soc_min ({inst.soc_min})")
                if soc > inst.soc_max + _EPS:
                    violations.append(f"{label}: SOC {momento} {trip.id} = {soc:.3f} > soc_max ({inst.soc_max})")
    return violations


def _check_initial_soc(sol: Solution, inst: Instance) -> list[str]:
    violations = []
    bus = inst.bus_type
    for idx, chain in enumerate(sol.chains):
        if not chain.trips or not chain.soc_before or recharge_events(chain, 0):
            continue
        first = chain.trips[0]
        km = 0.0 if chain.depot_id == first.start_station else inst.deadhead_km[(chain.depot_id, first.start_station)]
        max_soc = inst.soc_initial - km * bus.deadhead_kwh_per_km / bus.battery_kwh
        if chain.soc_before[0] > max_soc + _EPS:
            violations.append(
                f"{_etiqueta(idx, chain)}: SOC antes de {first.id} = {chain.soc_before[0]:.3f} supera el máximo "
                f"posible partiendo con soc_initial = {inst.soc_initial} ({max_soc:.3f})"
            )
    return violations


def _check_soc_consistency(sol: Solution, inst: Instance) -> list[str]:
    violations = []
    bus = inst.bus_type

    def km(i: str, j: str) -> float:
        return 0.0 if i == j else inst.deadhead_km[(i, j)]

    for idx, chain in enumerate(sol.chains):
        n = len(chain.trips)
        if not n or len(chain.soc_before) != n or len(chain.soc_after) != n:
            continue
        soc, lugar = inst.soc_initial, chain.depot_id
        for k, trip in enumerate(chain.trips):
            for carga in recharge_events(chain, k):
                soc -= km(lugar, carga.depot_id) * bus.deadhead_kwh_per_km / bus.battery_kwh
                soc = min(inst.soc_max, soc + carga.power_kw * (carga.end_time - carga.start_time) / 60 / bus.battery_kwh)
                lugar = carga.depot_id
            soc -= km(lugar, trip.start_station) * bus.deadhead_kwh_per_km / bus.battery_kwh
            esperado_antes = soc
            soc -= trip.distance_km * bus.trip_kwh_per_km / bus.battery_kwh
            for momento, guardado, esperado in (("antes de", chain.soc_before[k], esperado_antes),
                                                ("después de", chain.soc_after[k], soc)):
                if abs(guardado - esperado) > 1e-6:
                    violations.append(
                        f"{_etiqueta(idx, chain)}: SOC {momento} {trip.id} guardado = {guardado:.4f}, "
                        f"pero la dinámica da {esperado:.4f}"
                    )
                    break
            else:
                lugar = trip.end_station
                continue
            break
    return violations


def _check_recharge_positions(sol: Solution) -> list[str]:
    violations = []
    for idx, chain in enumerate(sol.chains):
        for k in sorted(chain.recharges):
            if k == 0:
                if any(e.power_kw > 0 and e.end_time > e.start_time for e in recharge_events(chain, 0)):
                    violations.append(f"{_etiqueta(idx, chain)}: carga antes del primer viaje (prohibida por la restricción (13))")
            elif k < 0 or k > len(chain.trips):
                violations.append(f"{_etiqueta(idx, chain)}: carga en la posición {k}, fuera de los {len(chain.trips)} viajes")
    return violations


def _check_return_soc(sol: Solution, inst: Instance) -> list[str]:
    violations = []
    bus = inst.bus_type

    def km(i: str, j: str) -> float:
        return 0.0 if i == j else inst.deadhead_km[(i, j)]

    for idx, chain in enumerate(sol.chains):
        if not chain.trips or not chain.soc_after or chain.return_depot_id is None:
            continue
        soc, lugar = chain.soc_after[-1], chain.trips[-1].end_station
        for carga_final in recharge_events(chain, len(chain.trips)):
            llegada = soc - km(lugar, carga_final.depot_id) * bus.deadhead_kwh_per_km / bus.battery_kwh
            if llegada < inst.soc_min - _EPS:
                violations.append(
                    f"{_etiqueta(idx, chain)}: llega a cargar a {carga_final.depot_id} con SOC {llegada:.3f} < soc_min ({inst.soc_min})"
                )
            soc = min(inst.soc_max, llegada + carga_final.power_kw * carga_final.duration / 60 / bus.battery_kwh)
            lugar = carga_final.depot_id
        llegada = soc - km(lugar, chain.return_depot_id) * bus.deadhead_kwh_per_km / bus.battery_kwh
        if llegada < inst.soc_min - _EPS:
            violations.append(
                f"{_etiqueta(idx, chain)}: llega al depósito {chain.return_depot_id} con SOC {llegada:.3f} < soc_min ({inst.soc_min})"
            )
    return violations


def _check_charging_resources(sol: Solution, inst: Instance) -> list[str]:
    violations = []
    by_depot: dict[str, list] = {}
    for chain in sol.chains:
        for recharge in all_recharge_events(chain):
            by_depot.setdefault(recharge.depot_id, []).append(recharge)

    depots = {d.id: d for d in inst.depots}
    for depot_id, events in by_depot.items():
        depot = depots.get(depot_id)
        if depot is None:
            violations.append(f"Carga en depósito desconocido: {depot_id}")
            continue
        time_points = sorted({e.start_time for e in events} | {e.end_time for e in events})
        for t in time_points[:-1]:
            active = [e for e in events if e.start_time <= t < e.end_time]
            if not active:
                continue
            if len(active) > depot.n_chargers:
                violations.append(
                    f"Depósito {depot_id} en t={t}: {len(active)} cargas simultáneas > "
                    f"{depot.n_chargers} cargadores"
                )
            total_power = sum(e.power_kw for e in active)
            if total_power > depot.power_kw + _EPS:
                violations.append(
                    f"Depósito {depot_id} en t={t}: potencia total {total_power:.1f} kW > {depot.power_kw} kW"
                )
    return violations


def _check_depot_capacity(sol: Solution, inst: Instance) -> list[str]:
    capacidades = {d.id: d.capacity for d in inst.depots if d.capacity is not None}
    if not capacidades:
        return []
    inicio, fin = inst.horizon_start, inst.horizon_end
    delta = {d: [0] * (fin - inicio + 2) for d in capacidades}

    def dentro(deposito: str, desde: int, hasta: int) -> None:
        if deposito not in delta:
            return
        desde, hasta = max(desde, inicio), min(hasta, fin + 1)
        if hasta > desde:
            delta[deposito][desde - inicio] += 1
            delta[deposito][hasta - inicio] -= 1

    def visitas(eventos) -> list[tuple[str, int, int]]:
        grupos: list[list] = []
        for e in eventos:
            if grupos and grupos[-1][0] == e.depot_id:
                grupos[-1][2] = max(grupos[-1][2], e.end_time)
            else:
                grupos.append([e.depot_id, e.start_time, e.end_time])
        return [tuple(g) for g in grupos]

    for chain in sol.chains:
        if not chain.trips:
            continue
        viajes = chain.trips
        primero, ultimo = viajes[0], viajes[-1]
        dentro(chain.depot_id, inicio, primero.start_time - inst.deadhead(chain.depot_id, primero.start_station))
        for k in range(1, len(viajes)):
            previo, siguiente = viajes[k - 1], viajes[k]
            grupos = visitas(recharge_events(chain, k))
            for i, (deposito, desde, hasta) in enumerate(grupos):
                if i == 0:
                    desde = previo.end_time + inst.deadhead(previo.end_station, deposito)
                if i == len(grupos) - 1:
                    hasta = siguiente.start_time - inst.deadhead(deposito, siguiente.start_station)
                dentro(deposito, desde, hasta)
        grupos = visitas(recharge_events(chain, len(viajes)))
        lugar, momento = ultimo.end_station, ultimo.end_time
        for i, (deposito, desde, hasta) in enumerate(grupos):
            if i == 0:
                desde = ultimo.end_time + inst.deadhead(ultimo.end_station, deposito)
            dentro(deposito, desde, hasta)
            lugar, momento = deposito, hasta
        if chain.return_depot_id is not None:
            llegada = momento + inst.deadhead(lugar, chain.return_depot_id)
            dentro(chain.return_depot_id, llegada, fin + 1)

    violations = []
    for deposito, capacidad in capacidades.items():
        ocupacion, maximo, cuando, minutos = 0, 0, inicio, 0
        for paso, cambio in enumerate(delta[deposito][:-1]):
            ocupacion += cambio
            minutos += ocupacion > capacidad
            if ocupacion > maximo:
                maximo, cuando = ocupacion, inicio + paso
        if maximo > capacidad:
            violations.append(f"Depósito {deposito}: hasta {maximo} buses estacionados a la vez (t={cuando}) > "
                              f"capacidad de {capacidad} ({minutos} minutos sobre la capacidad)")
    return violations


def _check_depot_balance(sol: Solution, inst: Instance) -> list[str]:
    violations = []
    starts = Counter(chain.depot_id for chain in sol.chains if chain.trips)
    returns = Counter(chain.return_depot_id for chain in sol.chains if chain.trips and chain.return_depot_id)
    for depot in inst.depots:
        out_count, in_count = starts.get(depot.id, 0), returns.get(depot.id, 0)
        if out_count != in_count:
            violations.append(
                f"Depósito {depot.id}: salieron {out_count} bus(es) y volvieron {in_count} (restricción (6))"
            )
    return violations
