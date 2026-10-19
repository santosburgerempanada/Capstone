from __future__ import annotations

import re

from ebus.data.fleet import deadhead_energy_soc, trip_energy_soc
from ebus.heuristics.depot_assignment import assign_return_depots, retorno_factible
from ebus.heuristics.destroy import quitar_hasta_factible
from ebus.heuristics.soc_adjustment import es_factible, recompute_soc
from ebus.model.cost import evaluate
from ebus.model.entities import Instance, Solution, Trip, TripChain

_EPS = 1e-9


def _gurobi():
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as error:
        raise RuntimeError(
            "La post-reparación necesita Gurobi (`gurobipy`) y no está disponible. Instálelo y configure la licencia "
            "(variable de entorno GRB_LICENSE_FILE)."
        ) from error
    return gp, GRB


def _cadena(deposito: str, viajes: list[Trip], retorno: str, inst: Instance) -> TripChain:
    cadena = TripChain(depot_id=deposito, trips=list(viajes), return_depot_id=retorno)
    recompute_soc(cadena, inst)
    return cadena


def _factible_solo(trip: Trip, inst: Instance) -> bool:
    deposito = inst.nearest_depot(trip.start_station).id
    return es_factible(_cadena(deposito, [trip], deposito, inst), inst)


def min_bus_chains(trips: list[Trip], inst: Instance, time_limit_s: float,
                   cupos: dict[str, int] | None = None) -> list[TripChain]:
    if not trips:
        return []
    sueltos = [t for t in trips if not _factible_solo(t, inst)]
    viajes = sorted((t for t in trips if t not in sueltos), key=lambda t: (t.start_time, t.id))
    cadenas = [_cadena(inst.nearest_depot(t.start_station).id, [t], inst.nearest_depot(t.start_station).id, inst)
               for t in sorted(sueltos, key=lambda t: t.id)]
    if not viajes:
        return cadenas

    gp, GRB = _gurobi()
    bus = inst.bus_type
    capacidad = inst.soc_initial - inst.soc_min
    depositos = [d.id for d in inst.depots]
    precio_noche = inst.tariff.overnight_price_per_kwh
    lugares = {t.start_station for t in viajes} | {t.end_station for t in viajes} | set(depositos)

    def costo_deadhead(i: str, j: str) -> float:
        km = inst.deadhead_distance_km(i, j)
        return km * inst.deadhead_cost_per_km + km * bus.deadhead_kwh_per_km * precio_noche

    def e_deadhead(i: str, j: str) -> float:
        return deadhead_energy_soc(inst.deadhead_distance_km(i, j), bus)

    salidas = {(d, i): t for d in depositos for i, t in enumerate(viajes)
               if inst.horizon_start + inst.deadhead(d, t.start_station) <= t.start_time}
    regresos = {(i, d): t for i, t in enumerate(viajes) for d in depositos
                if t.end_time + inst.deadhead(t.end_station, d) <= inst.horizon_end}
    conexiones = {(i, j) for i, a in enumerate(viajes) for j, b in enumerate(viajes) if i != j and inst.can_connect(a, b)}

    m = gp.Model("post_reparacion")
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = time_limit_s
    m.Params.MIPGap = 0.0
    s = {k: m.addVar(vtype=GRB.BINARY) for k in salidas}
    r = {k: m.addVar(vtype=GRB.BINARY) for k in regresos}
    x = {k: m.addVar(vtype=GRB.BINARY) for k in conexiones}
    u = {i: m.addVar(lb=0.0, ub=capacidad) for i in range(len(viajes))}

    for i in range(len(viajes)):
        m.addConstr(gp.quicksum(s[d, i] for d in depositos if (d, i) in s)
                    + gp.quicksum(x[j, i] for j in range(len(viajes)) if (j, i) in x) == 1)
        m.addConstr(gp.quicksum(r[i, d] for d in depositos if (i, d) in r)
                    + gp.quicksum(x[i, j] for j in range(len(viajes)) if (i, j) in x) == 1)
    for d in depositos:
        m.addConstr(gp.quicksum(v for (dd, _), v in s.items() if dd == d) == gp.quicksum(v for (_, dd), v in r.items() if dd == d))
    for d, cupo in (cupos or {}).items():
        m.addConstr(gp.quicksum(v for (dd, _), v in s.items() if dd == d) <= max(cupo, 0), name=f"cupo[{d}]")

    grande = capacidad + max(trip_energy_soc(t, bus) for t in viajes) + 2 * max(
        e_deadhead(a, b) for a in lugares for b in lugares) + 1.0
    for (d, i), t in salidas.items():
        m.addConstr(u[i] >= e_deadhead(d, t.start_station) + trip_energy_soc(t, bus) - grande * (1 - s[d, i]))
    for i, j in conexiones:
        a, b = viajes[i], viajes[j]
        m.addConstr(u[j] >= u[i] + e_deadhead(a.end_station, b.start_station) + trip_energy_soc(b, bus)
                    - grande * (1 - x[i, j]))
    for (i, d), t in regresos.items():
        m.addConstr(u[i] + e_deadhead(t.end_station, d) <= capacidad + grande * (1 - r[i, d]))

    por_bus = (inst.soc_max - inst.soc_initial) * bus.battery_kwh * precio_noche
    costo = (gp.quicksum((por_bus + costo_deadhead(d, viajes[i].start_station)) * v for (d, i), v in s.items())
             + gp.quicksum(costo_deadhead(viajes[i].end_station, d) * v for (i, d), v in r.items())
             + gp.quicksum(costo_deadhead(viajes[i].end_station, viajes[j].start_station) * v for (i, j), v in x.items()))
    cota = len(viajes) * (por_bus + 3 * max(costo_deadhead(a, b) for a in lugares for b in lugares)) + 1.0
    m.setObjective(cota * gp.quicksum(s.values()) + costo, GRB.MINIMIZE)
    m.optimize()
    if m.SolCount == 0:
        raise RuntimeError(f"La post-reparación no encontró solución (estado de Gurobi {m.Status})")

    siguiente = {i: j for (i, j), v in x.items() if v.X > 0.5}
    for (d, i), v in s.items():
        if v.X > 0.5:
            recorrido = [i]
            while recorrido[-1] in siguiente:
                recorrido.append(siguiente[recorrido[-1]])
            retorno = next(dd for (ii, dd), w in r.items() if ii == recorrido[-1] and w.X > 0.5)
            cadenas.append(_cadena(d, [viajes[k] for k in recorrido], retorno, inst))
    return cadenas


def _numero(bus_id: str | None) -> int:
    coincidencia = re.fullmatch(r"B(\d+)", bus_id or "")
    return int(coincidencia.group(1)) if coincidencia else 0


def _liberar_retornos_infactibles(sol: Solution, inst: Instance) -> tuple[bool, list[Trip]]:
    cambio = False
    quitados: list[Trip] = []
    if len(inst.depots) == 1:
        return cambio, quitados
    cadenas: list[TripChain] = []
    for cadena in sol.chains:
        if not cadena.trips or retorno_factible(cadena, cadena.return_depot_id, inst):
            cadenas.append(cadena)
            continue
        cambio = True
        propia = cadena.copy()
        propia.return_depot_id = propia.depot_id
        recompute_soc(propia, inst)
        resto, mas = quitar_hasta_factible([propia], len(propia.trips) - 1, inst)
        cadenas += resto
        quitados += mas
    sol.chains = cadenas
    return cambio, quitados


def _cupos(sol: Solution, inst: Instance) -> dict[str, int]:
    salen: dict[str, int] = {}
    for cadena in sol.chains:
        salen[cadena.depot_id] = salen.get(cadena.depot_id, 0) + 1
    return {d.id: d.capacity - salen.get(d.id, 0) for d in inst.depots if d.capacity is not None}


def _buses_nuevos(pendientes: list[Trip], sol: Solution, inst: Instance, time_limit_s: float) -> list[TripChain]:
    cupos = _cupos(sol, inst)
    if not cupos or not pendientes:
        return min_bus_chains(pendientes, inst, time_limit_s)
    try:
        return min_bus_chains(pendientes, inst, time_limit_s, cupos)
    except RuntimeError:
        return min_bus_chains(pendientes, inst, time_limit_s)


def cerrar_solucion(sol: Solution, pendientes: list[Trip], inst: Instance, time_limit_s: float) -> Solution:
    cerrada = sol.copy()
    cerrada.unassigned = []
    pendientes = list(pendientes)
    for _ in range(len(cerrada.chains) + len(pendientes) + 1):
        nuevas = _buses_nuevos(pendientes, cerrada, inst, time_limit_s)
        siguiente = max((_numero(c.bus_id) for c in cerrada.chains), default=0) + 1
        for k, cadena in enumerate(nuevas):
            cadena.bus_id = f"B{siguiente + k:03d}"
        cerrada.chains.extend(nuevas)
        assign_return_depots(cerrada, inst)
        cambio, pendientes = _liberar_retornos_infactibles(cerrada, inst)
        if not cambio:
            break
    cerrada.cost = evaluate(cerrada, inst).total
    return cerrada
