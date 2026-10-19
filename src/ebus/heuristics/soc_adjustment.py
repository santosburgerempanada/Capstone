from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import replace

from ebus.data.fleet import deadhead_energy_soc, trip_energy_soc
from ebus.model.entities import Instance, RechargeEvent, Trip, TripChain, all_recharge_events, recharge_events

_EPS = 1e-9
_MAX_AJUSTES = 8

contador_escenarios: Counter = Counter()


def _e_deadhead(inst: Instance, i: str, j: str) -> float:
    return deadhead_energy_soc(inst.deadhead_distance_km(i, j), inst.bus_type)


def energy_needed(origen: str, c: Trip, destino: str, inst: Instance) -> float:
    return trip_energy_soc(c, inst.bus_type) + _e_deadhead(inst, origen, c.start_station) + _e_deadhead(
        inst, c.end_station, destino)


def recompute_soc(chain: TripChain, inst: Instance) -> None:
    bus = inst.bus_type
    soc = inst.soc_initial
    ubicacion = chain.depot_id
    antes: list[float] = []
    despues: list[float] = []
    for k, trip in enumerate(chain.trips):
        for carga in recharge_events(chain, k):
            soc -= deadhead_energy_soc(inst.deadhead_distance_km(ubicacion, carga.depot_id), bus)
            soc = min(inst.soc_max, soc + carga.power_kw * carga.duration / 60 / bus.battery_kwh)
            ubicacion = carga.depot_id
        soc -= deadhead_energy_soc(inst.deadhead_distance_km(ubicacion, trip.start_station), bus)
        antes.append(soc)
        soc -= trip_energy_soc(trip, bus)
        despues.append(soc)
        ubicacion = trip.end_station
    chain.soc_before[:] = antes
    chain.soc_after[:] = despues


def _retorno(chain: TripChain) -> str:
    return chain.return_depot_id or chain.depot_id


def evaluar_cadena(chain: TripChain, inst: Instance) -> tuple[float, bool]:
    bus = inst.bus_type
    soc, lugar, t = inst.soc_initial, chain.depot_id, inst.horizon_start
    minimo, ok = soc, True
    n = len(chain.trips)
    for k in range(n + 1):
        for carga in recharge_events(chain, k):
            if k == 0 and carga.power_kw > 0 and carga.duration > 0:
                ok = False
            if carga.start_time < t + inst.deadhead(lugar, carga.depot_id) or carga.end_time < carga.start_time:
                ok = False
            soc -= _e_deadhead(inst, lugar, carga.depot_id)
            minimo = min(minimo, soc)
            soc = min(inst.soc_max, soc + carga.power_kw * carga.duration / 60 / bus.battery_kwh)
            lugar, t = carga.depot_id, carga.end_time
        if k == n:
            break
        viaje = chain.trips[k]
        if viaje.start_time < t + inst.deadhead(lugar, viaje.start_station):
            ok = False
        soc -= _e_deadhead(inst, lugar, viaje.start_station)
        minimo = min(minimo, soc)
        soc -= trip_energy_soc(viaje, bus)
        minimo = min(minimo, soc)
        lugar, t = viaje.end_station, viaje.end_time
    retorno = _retorno(chain)
    if t + inst.deadhead(lugar, retorno) > inst.horizon_end:
        ok = False
    minimo = min(minimo, soc - _e_deadhead(inst, lugar, retorno))
    return minimo, ok


def es_factible(chain: TripChain, inst: Instance) -> bool:
    minimo, ok = evaluar_cadena(chain, inst)
    return ok and minimo >= inst.soc_min - _EPS


def _soc_al_volver(chain: TripChain, inst: Instance) -> float:
    bus = inst.bus_type
    if not chain.trips:
        return inst.soc_initial - _e_deadhead(inst, chain.depot_id, _retorno(chain))
    soc, lugar = chain.soc_after[-1], chain.trips[-1].end_station
    for carga in recharge_events(chain, len(chain.trips)):
        soc -= _e_deadhead(inst, lugar, carga.depot_id)
        soc = min(inst.soc_max, soc + carga.power_kw * carga.duration / 60 / bus.battery_kwh)
        lugar = carga.depot_id
    return soc - _e_deadhead(inst, lugar, _retorno(chain))


def _tiene_carga(chain: TripChain, k: int) -> bool:
    return any(e.power_kw > 0 and e.duration > 0 for e in recharge_events(chain, k))


def _insertar(chain: TripChain, pos: int, c: Trip, inst: Instance, quitar_cargas: bool = False) -> TripChain:
    nueva = chain.copy()
    recargas = {}
    for k, eventos in nueva.recharges.items():
        if k < pos:
            recargas[k] = eventos
        elif k > pos:
            recargas[k + 1] = eventos
    if not quitar_cargas:
        antes, despues = [], []
        for e in recharge_events(nueva, pos):
            if e.end_time + inst.deadhead(e.depot_id, c.start_station) <= c.start_time:
                antes.append(e)
            elif c.end_time + inst.deadhead(c.end_station, e.depot_id) <= e.start_time:
                despues.append(e)
        if antes:
            recargas[pos] = antes
        if despues:
            recargas[pos + 1] = despues
    nueva.trips = chain.trips[:pos] + [c] + chain.trips[pos:]
    nueva.recharges = recargas
    recompute_soc(nueva, inst)
    return nueva


def clasificar_insercion(chain: TripChain, pos: int, c: Trip, inst: Instance) -> tuple[str, str, float]:
    base = chain.copy()
    recompute_soc(base, inst)
    n = len(base.trips)
    origen = base.trips[pos - 1].end_station if pos > 0 else base.depot_id
    destino = base.trips[pos].start_station if pos < n else _retorno(base)
    a2 = base.soc_after[pos - 1] if pos > 0 else inst.soc_initial
    soc_vuelta = _soc_al_volver(base, inst)
    if pos < n:
        b1 = base.soc_before[pos]
        e_lowest = min(base.soc_before[pos:] + base.soc_after[pos:] + [soc_vuelta])
    else:
        b1 = e_lowest = soc_vuelta
    e_need = energy_needed(origen, c, destino, inst)

    if not _tiene_carga(base, pos):
        e_add = e_need + (b1 - a2)
        e_available = e_lowest - inst.soc_min
        if e_add <= _EPS:
            return "I", "S1", 0.0
        if e_available >= e_add - _EPS:
            return "I", "S2", 0.0
        return "I", "S3", e_add - e_available
    e_available = e_lowest - inst.soc_min - (b1 - a2)
    if e_available >= e_need - _EPS:
        return "II", "S1", 0.0
    return "II", "S2", e_need - e_available


def _potencia(inst: Instance, depot_id: str) -> float:
    deposito = next(d for d in inst.depots if d.id == depot_id)
    return min(inst.bus_type.max_charge_kw, deposito.power_kw)


def _siguiente(chain: TripChain, k: int, inst: Instance) -> tuple[str, int]:
    if k < len(chain.trips):
        return chain.trips[k].start_station, chain.trips[k].start_time
    return _retorno(chain), inst.horizon_end


def _con_carga_nueva(chain: TripChain, k: int, minutos: int, inst: Instance) -> TripChain | None:
    if minutos <= 0:
        return None
    previo = chain.trips[k - 1]
    deposito = inst.nearest_depot(previo.end_station)
    lugar, t_siguiente = _siguiente(chain, k, inst)
    llegada = previo.end_time + inst.deadhead(previo.end_station, deposito.id)
    if llegada + minutos + inst.deadhead(deposito.id, lugar) > t_siguiente:
        return None
    nueva = chain.copy()
    nueva.recharges[k] = [RechargeEvent(deposito.id, llegada, llegada + minutos, _potencia(inst, deposito.id))]
    recompute_soc(nueva, inst)
    return nueva


def _con_carga_extendida(chain: TripChain, k: int, minutos: int, inst: Instance) -> TripChain | None:
    if minutos <= 0:
        return None
    nueva = chain.copy()
    eventos = list(recharge_events(nueva, k))
    ultimo = eventos[-1]
    if ultimo.power_kw > 0 and ultimo.duration > 0:
        extendido = replace(ultimo, end_time=ultimo.end_time + minutos)
    else:
        extendido = replace(ultimo, power_kw=_potencia(inst, ultimo.depot_id),
                            end_time=max(ultimo.end_time, ultimo.start_time + minutos))
    lugar, t_siguiente = _siguiente(nueva, k, inst)
    if extendido.end_time + inst.deadhead(extendido.depot_id, lugar) > t_siguiente:
        return None
    eventos[-1] = extendido
    nueva.recharges[k] = eventos
    recompute_soc(nueva, inst)
    return nueva


def _ajustar(construir: Callable[[int], TripChain | None], potencia: float, deficit: float,
             inst: Instance) -> TripChain | None:
    bateria = inst.bus_type.battery_kwh
    minutos = max(1, math.ceil(deficit * bateria / potencia * 60 - _EPS))
    for _ in range(_MAX_AJUSTES):
        candidata = construir(minutos)
        if candidata is None:
            return None
        minimo, horarios_ok = evaluar_cadena(candidata, inst)
        if not horarios_ok:
            return None
        if minimo >= inst.soc_min - _EPS:
            return candidata
        minutos += max(1, math.ceil((inst.soc_min - minimo) * bateria / potencia * 60 - _EPS))
    return None


def _intentar_extender(chain: TripChain, k: int, deficit: float, inst: Instance) -> TripChain | None:
    if not 1 <= k <= len(chain.trips) or not recharge_events(chain, k):
        return None
    ultimo = recharge_events(chain, k)[-1]
    potencia = ultimo.power_kw if ultimo.power_kw > 0 and ultimo.duration > 0 else _potencia(inst, ultimo.depot_id)
    return _ajustar(lambda m: _con_carga_extendida(chain, k, m, inst), potencia, deficit, inst)


def _intentar_nueva(chain: TripChain, k: int, deficit: float, inst: Instance) -> TripChain | None:
    if not 1 <= k <= len(chain.trips) or recharge_events(chain, k):
        return None
    previo = chain.trips[k - 1]
    deposito = inst.nearest_depot(previo.end_station)
    lugar, _ = _siguiente(chain, k, inst)
    desvio = (_e_deadhead(inst, previo.end_station, deposito.id) + _e_deadhead(inst, deposito.id, lugar)
              - _e_deadhead(inst, previo.end_station, lugar))
    return _ajustar(lambda m: _con_carga_nueva(chain, k, m, inst), _potencia(inst, deposito.id),
                    deficit + desvio, inst)


def recharge_opportunity(chain: TripChain, pos: int, deficit: float, inst: Instance) -> TripChain | None:
    n = len(chain.trips)
    deficit = max(deficit, 0.0)
    intentos = [lambda: _intentar_extender(chain, pos, deficit, inst),
                lambda: _intentar_nueva(chain, pos, deficit, inst),
                lambda: _intentar_extender(chain, pos + 1, deficit, inst),
                lambda: _intentar_nueva(chain, pos + 1, deficit, inst)]
    intentos += [lambda k=k: _intentar_extender(chain, k, deficit, inst) for k in range(pos - 1, 0, -1)]
    intentos += [lambda k=k: _intentar_nueva(chain, k, deficit, inst) for k in range(pos - 1, 0, -1)]
    intentos += [lambda k=k: _intentar_nueva(chain, k, deficit, inst) for k in range(pos + 2, n + 1)]
    for intento in intentos:
        resultado = intento()
        if resultado is not None:
            return resultado
    return None


def adjust_case_one(chain: TripChain, pos: int, c: Trip, inst: Instance) -> TripChain | None:
    candidata = _insertar(chain, pos, c, inst)
    minimo, horarios_ok = evaluar_cadena(candidata, inst)
    if not horarios_ok:
        return None
    if minimo >= inst.soc_min - _EPS:
        return candidata
    return recharge_opportunity(candidata, pos, inst.soc_min - minimo, inst)


def adjust_case_two(chain: TripChain, pos: int, c: Trip, inst: Instance) -> TripChain | None:
    sin_carga = _insertar(chain, pos, c, inst, quitar_cargas=True)
    if es_factible(sin_carga, inst):
        return sin_carga
    con_carga = _insertar(chain, pos, c, inst)
    minimo, horarios_ok = evaluar_cadena(con_carga, inst)
    if not horarios_ok:
        return None
    if minimo >= inst.soc_min - _EPS:
        return con_carga
    deficit = inst.soc_min - minimo
    for k in (pos, pos + 1):
        resultado = _intentar_extender(con_carga, k, deficit, inst)
        if resultado is not None:
            return resultado
    return recharge_opportunity(con_carga, pos, deficit, inst)


def adjust_for_insertion(chain: TripChain, pos: int, c: Trip, inst: Instance) -> TripChain | None:
    n = len(chain.trips)
    if not 0 <= pos <= n:
        raise ValueError(f"posición {pos} fuera de la cadena de {n} viajes")
    if any(t.id == c.id for t in chain.trips):
        raise ValueError(f"el viaje {c.id} ya está en la cadena")
    if pos > 0:
        lugar, t = chain.trips[pos - 1].end_station, chain.trips[pos - 1].end_time
    else:
        lugar, t = chain.depot_id, inst.horizon_start
    if c.start_time < t + inst.deadhead(lugar, c.start_station):
        return None
    destino, t_destino = _siguiente(chain, pos, inst)
    if c.end_time + inst.deadhead(c.end_station, destino) > t_destino:
        return None
    caso, escenario, _ = clasificar_insercion(chain, pos, c, inst)
    resultado = adjust_case_two(chain, pos, c, inst) if caso == "II" else adjust_case_one(chain, pos, c, inst)
    if resultado is None:
        contador_escenarios[(caso, escenario, "imposible")] += 1
    else:
        cargado = sum(e.power_kw * e.duration for e in all_recharge_events(resultado))
        cargado_antes = sum(e.power_kw * e.duration for e in all_recharge_events(chain))
        contador_escenarios[(caso, escenario, "con carga" if cargado > cargado_antes + _EPS else "sin carga")] += 1
    return resultado
