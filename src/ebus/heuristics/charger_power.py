from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from ebus.heuristics.destroy import quitar_hasta_factible, quitar_viajes
from ebus.heuristics.soc_adjustment import _siguiente, es_factible, recompute_soc
from ebus.model.entities import Depot, Instance, RechargeEvent, Solution, Trip, TripChain, recharge_events

_EPS = 1e-9


@dataclass
class _Carga:

    cadena: int
    k: int
    j: int
    evento: RechargeEvent


def charging_profile(sol: Solution, inst: Instance) -> dict[str, list[_Carga]]:
    perfil: dict[str, list[_Carga]] = {d.id: [] for d in inst.depots}
    for i, cadena in enumerate(sol.chains):
        for k in sorted(cadena.recharges):
            for j, evento in enumerate(recharge_events(cadena, k)):
                if evento.duration > 0 and evento.flag == 0:
                    perfil.setdefault(evento.depot_id, []).append(_Carga(i, k, j, evento))
    return perfil


def _activas(cargas: list[_Carga], t: int, excluir: _Carga | None = None) -> list[_Carga]:
    return [c for c in cargas if c is not excluir and c.evento.start_time <= t < c.evento.end_time]


def _primera_violacion(cargas: list[_Carga], deposito: Depot) -> tuple[int, str] | None:
    for t in sorted({c.evento.start_time for c in cargas}):
        activas = _activas(cargas, t)
        if len(activas) > deposito.n_chargers:
            return t, "cargadores"
        if sum(c.evento.power_kw for c in activas) > deposito.power_kw + _EPS:
            return t, "potencia"
    return None


def _libre(cargas: list[_Carga], tramos: list[tuple[int, int]], potencia: float, deposito: Depot,
           excluir: _Carga) -> bool:
    for inicio, fin in tramos:
        if fin <= inicio:
            continue
        puntos = {inicio} | {c.evento.start_time for c in cargas if inicio < c.evento.start_time < fin}
        for t in puntos:
            activas = _activas(cargas, t, excluir)
            if len(activas) + 1 > deposito.n_chargers:
                return False
            if sum(c.evento.power_kw for c in activas) + potencia > deposito.power_kw + _EPS:
                return False
    return True


def _ventana(cadena: TripChain, k: int, j: int, inst: Instance) -> tuple[int, int]:
    eventos = recharge_events(cadena, k)
    evento = eventos[j]
    if j > 0:
        previo = eventos[j - 1]
        desde = previo.end_time + inst.deadhead(previo.depot_id, evento.depot_id)
    elif k > 0:
        viaje = cadena.trips[k - 1]
        desde = viaje.end_time + inst.deadhead(viaje.end_station, evento.depot_id)
    else:
        desde = inst.horizon_start + inst.deadhead(cadena.depot_id, evento.depot_id)
    if j + 1 < len(eventos):
        siguiente = eventos[j + 1]
        hasta = siguiente.start_time - inst.deadhead(evento.depot_id, siguiente.depot_id)
    else:
        lugar, t_siguiente = _siguiente(cadena, k, inst)
        hasta = t_siguiente - inst.deadhead(evento.depot_id, lugar)
    return desde, hasta


def _con_evento(cadena: TripChain, k: int, j: int, nuevo: RechargeEvent, inst: Instance) -> TripChain:
    copia = cadena.copy()
    eventos = list(recharge_events(copia, k))
    eventos[j] = nuevo
    copia.recharges[k] = eventos
    recompute_soc(copia, inst)
    return copia


def _aplicar(sol: Solution, carga: _Carga, nuevo: RechargeEvent, inst: Instance, revisar: bool = True) -> bool:
    cadena = _con_evento(sol.chains[carga.cadena], carga.k, carga.j, nuevo, inst)
    if revisar and not es_factible(cadena, inst):
        return False
    sol.chains[carga.cadena] = cadena
    carga.evento = nuevo
    return True


def _desplazar(sol: Solution, cargas: list[_Carga], carga: _Carga, deposito: Depot, inst: Instance) -> bool:
    evento = carga.evento
    duracion = evento.duration
    desde, hasta = _ventana(sol.chains[carga.cadena], carga.k, carga.j, inst)
    otros = [c for c in cargas if c is not carga]
    candidatos = {desde, hasta - duracion}
    candidatos |= {c.evento.end_time for c in otros} | {c.evento.start_time - duracion for c in otros}
    candidatos = sorted((s for s in candidatos if desde <= s <= hasta - duracion and s != evento.start_time),
                        key=lambda s: (abs(s - evento.start_time), s))
    for inicio in candidatos:
        if not _libre(cargas, [(inicio, inicio + duracion)], evento.power_kw, deposito, carga):
            continue
        if _aplicar(sol, carga, replace(evento, start_time=inicio, end_time=inicio + duracion), inst):
            return True
    return False


def _bajar_potencia(sol: Solution, cargas: list[_Carga], carga: _Carga, potencia: float, deposito: Depot,
                    inst: Instance) -> bool:
    evento = carga.evento
    duracion = math.ceil(evento.power_kw * evento.duration / potencia - _EPS)
    desde, hasta = _ventana(sol.chains[carga.cadena], carga.k, carga.j, inst)
    inicio = min(evento.start_time, hasta - duracion)
    if inicio < desde:
        return False
    fin = inicio + duracion
    tramos = [(inicio, evento.start_time), (evento.end_time, fin)]
    if not _libre(cargas, tramos, potencia, deposito, carga):
        return False
    return _aplicar(sol, carga, replace(evento, start_time=inicio, end_time=fin, power_kw=potencia), inst)


def _marcar(sol: Solution, cargas: list[_Carga], carga: _Carga, flag: int, inst: Instance) -> None:
    _aplicar(sol, carga, replace(carga.evento, flag=flag), inst, revisar=False)
    cargas.remove(carga)


def _resolver_cargadores(sol: Solution, cargas: list[_Carga], t: int, deposito: Depot, inst: Instance,
                         rng: np.random.Generator) -> None:
    activas = _activas(cargas, t)
    for indice in rng.permutation(len(activas)):
        if len(_activas(cargas, t)) <= deposito.n_chargers:
            return
        _desplazar(sol, cargas, activas[indice], deposito, inst)
    activas = _activas(cargas, t)
    exceso = len(activas) - deposito.n_chargers
    if exceso > 0:
        for indice in rng.choice(len(activas), size=exceso, replace=False):
            _marcar(sol, cargas, activas[indice], 1, inst)


def _resolver_potencia(sol: Solution, cargas: list[_Carga], t: int, deposito: Depot, inst: Instance,
                       rng: np.random.Generator) -> None:
    activas = _activas(cargas, t)
    media = deposito.power_kw / len(activas)
    altas = [c for c in activas if c.evento.power_kw > media + _EPS]
    for indice in rng.permutation(len(altas)):
        if sum(c.evento.power_kw for c in _activas(cargas, t)) <= deposito.power_kw + _EPS:
            return
        if not _bajar_potencia(sol, cargas, altas[indice], media, deposito, inst):
            _marcar(sol, cargas, altas[indice], 2, inst)


def fine_tune(sol: Solution, inst: Instance, rng: np.random.Generator) -> Solution:
    nueva = Solution(chains=list(sol.chains), unassigned=list(sol.unassigned), cost=None)
    depositos = {d.id: d for d in inst.depots}
    for depot_id, cargas in charging_profile(nueva, inst).items():
        deposito = depositos[depot_id]
        while (violacion := _primera_violacion(cargas, deposito)) is not None:
            t, tipo = violacion
            if tipo == "cargadores":
                _resolver_cargadores(nueva, cargas, t, deposito, inst, rng)
            else:
                _resolver_potencia(nueva, cargas, t, deposito, inst, rng)
    return nueva


def _viajes_a_quitar(cadena: TripChain, rng: np.random.Generator) -> list[Trip]:
    n = len(cadena.trips)
    con_carga = sorted(k for k in cadena.recharges
                       if any(e.duration > 0 and e.flag == 0 for e in recharge_events(cadena, k)))
    quitar: list[Trip] = []
    for k in sorted(cadena.recharges):
        for evento in recharge_events(cadena, k):
            if evento.flag == 1 and k < n:
                quitar.append(cadena.trips[k])
            elif evento.flag == 2:
                siguiente = min((kk for kk in con_carga if kk > k), default=n)
                entre = cadena.trips[k:siguiente]
                if entre:
                    quitar.append(entre[rng.integers(len(entre))])
    return list({t.id: t for t in quitar}.values())


def remove_flagged_trips(sol: Solution, inst: Instance, rng: np.random.Generator) -> tuple[Solution, list[Trip]]:
    cadenas: list[TripChain] = []
    quitados: list[Trip] = []
    for cadena in sol.chains:
        if not any(e.flag for k in cadena.recharges for e in recharge_events(cadena, k)):
            cadenas.append(cadena)
            continue
        quitar = _viajes_a_quitar(cadena, rng)
        limpia = cadena.copy()
        limpia.recharges = {k: eventos for k in cadena.recharges
                            if (eventos := [replace(e) for e in recharge_events(cadena, k) if not e.flag])}
        ids = {t.id for t in quitar}
        primera = min((i for i, t in enumerate(cadena.trips) if t.id in ids), default=len(cadena.trips))
        actual = quitar_viajes(Solution(chains=[limpia]), quitar, inst).chains
        actual, mas = quitar_hasta_factible(actual, primera, inst)
        quitados += quitar + mas
        cadenas += actual
    return Solution(chains=cadenas, unassigned=list(sol.unassigned), cost=None), quitados


def enforce_charging_resources(sol: Solution, inst: Instance,
                               rng: np.random.Generator) -> tuple[Solution, list[Trip]]:
    return remove_flagged_trips(fine_tune(sol, inst, rng), inst, rng)
