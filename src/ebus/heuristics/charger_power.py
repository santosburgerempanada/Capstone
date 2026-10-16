"""Sub-rutina de ajuste de cargadores y potencia (Sección 3.2.2, Fig. 5 del paper).

El repair ajusta la batería de un bus a la vez y no mira cuántos otros buses cargan al mismo tiempo en el depósito. Esta
sub-rutina corre después del repair y antes de la post-reparación, y deja la solución dentro de los límites de cada
depósito: a lo más `C_d` cargas simultáneas (restricción (7)) y a lo más `P_d` kW en total (restricción (8)).

1. **Ajuste fino** (`fine_tune`). Recorre los instantes en que un depósito excede sus límites, del más temprano al más
   tardío:
   - Más cargas que cargadores: intenta correr cargas a otro horario. Si no se puede, marca al azar las que sobran
     (`flag = 1`).
   - Más potencia que `P_d`: baja la potencia de las cargas activas a la media `P_d / |R_dt|` y alarga su duración para
     entregar la misma energía. La carga que no puede alargarse se marca (`flag = 2`).
2. **Quitar viajes** (`remove_flagged_trips`). Con `flag = 1` se quita el viaje que sigue a la carga; con `flag = 2`, un
   viaje al azar entre esa carga y la siguiente. Los viajes quitados van a la post-reparación.

Decisiones del equipo (ver docs/decisiones/alns.md, Paso 7):
- (a) Los límites se revisan **en tiempo continuo** (minutos enteros), como el validador, y no en periodos de 15 minutos:
  así una solución ajustada nunca tiene un conflicto que el validador sí vea.
- (b) Una carga se corre, con la misma duración y potencia, al horario libre **más cercano** dentro del tiempo en que el
  bus está detenido en el depósito, y solo si la batería del bus sigue siendo factible. "Libre" significa que ahí no
  excede ni los cargadores ni la potencia.
- (c) Las cargas que sobran se marcan al azar, con el `rng` de la librería.
- (d) La carga marcada **se elimina** junto con el viaje que indica el paper. Si sin esa carga la cadena se queda sin
  batería, se quitan también los viajes siguientes hasta que vuelva a ser factible.
- Las cargas que se alargan por potencia mantienen su energía (minutos redondeados hacia arriba). Nada de lo que se cambia
  crea un conflicto nuevo: por eso el ajuste siempre termina y la solución que entrega cumple los límites."""

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
    """Una carga de la solución y dónde está: cadena `cadena`, intervalo `k` (entre los viajes `k-1` y `k`), segmento
    `j` de ese intervalo."""

    cadena: int
    k: int
    j: int
    evento: RechargeEvent


def charging_profile(sol: Solution, inst: Instance) -> dict[str, list[_Carga]]:
    """Cargas de la solución agrupadas por depósito. `R_dt` (las activas en el instante `t`) sale de aquí con `_activas`.
    Los marcadores de visita sin duración no ocupan cargador y no se incluyen; las cargas ya marcadas tampoco."""
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
    """Primer instante en que el depósito excede sus cargadores o su potencia, y cuál de los dos. El uso solo sube cuando
    empieza una carga, así que basta revisar los inicios."""
    for t in sorted({c.evento.start_time for c in cargas}):
        activas = _activas(cargas, t)
        if len(activas) > deposito.n_chargers:
            return t, "cargadores"
        if sum(c.evento.power_kw for c in activas) > deposito.power_kw + _EPS:
            return t, "potencia"
    return None


def _libre(cargas: list[_Carga], tramos: list[tuple[int, int]], potencia: float, deposito: Depot,
           excluir: _Carga) -> bool:
    """Una carga de `potencia` kW cabe en los `tramos` sin exceder los cargadores ni la potencia del depósito (sin contar
    la carga `excluir`, que es la que se mueve)."""
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
    """Desde cuándo y hasta cuándo puede estar cargando el bus en el segmento `j` del intervalo `k`: entre que llega al
    depósito (desde el viaje o el segmento anterior) y la hora en que debe salir hacia lo que sigue."""
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
    """Copia de la cadena con el segmento `j` del intervalo `k` reemplazado por `nuevo`, y el SOC recalculado."""
    copia = cadena.copy()
    eventos = list(recharge_events(copia, k))
    eventos[j] = nuevo
    copia.recharges[k] = eventos
    recompute_soc(copia, inst)
    return copia


def _aplicar(sol: Solution, carga: _Carga, nuevo: RechargeEvent, inst: Instance, revisar: bool = True) -> bool:
    """Reemplaza la carga en la solución si la cadena sigue siendo factible (o sin revisar, si `revisar=False`)."""
    cadena = _con_evento(sol.chains[carga.cadena], carga.k, carga.j, nuevo, inst)
    if revisar and not es_factible(cadena, inst):
        return False
    sol.chains[carga.cadena] = cadena
    carga.evento = nuevo
    return True


def _desplazar(sol: Solution, cargas: list[_Carga], carga: _Carga, deposito: Depot, inst: Instance) -> bool:
    """Decisión (b): corre la carga, con su duración y potencia, al horario libre más cercano dentro de su ventana.
    Basta probar los bordes de la ventana y los bordes de las otras cargas: si hay un horario libre, deslizándolo
    hacia un lado se llega a uno de ellos sin dejar de estar libre."""
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
    """Baja la carga a `potencia` kW y la alarga para entregar la misma energía: primero hacia después; si no cabe, también
    hacia antes, dentro de su ventana. Los minutos agregados deben quedar libres en el depósito (en los que ya ocupaba, la
    potencia solo baja)."""
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
    """Marca la carga y la saca del perfil del depósito: ya no cuenta (la elimina `remove_flagged_trips`)."""
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
    """Fase 1. Copia de `sol` dentro de los límites de cargadores y potencia de cada depósito, contando solo las cargas
    sin marcar. Las cargas que no se pudieron arreglar quedan en sus cadenas con `flag` 1 (faltan cargadores) o 2 (falta
    potencia). Cada arreglo elimina el conflicto en su instante sin crear otros, así que el ciclo termina.

    No modifica `sol`: cada cadena que cambia se reemplaza por una copia, y las demás se comparten (no se copian las
    cientos de cadenas de una instancia real en cada iteración)."""
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
    """Flag 1: el viaje que sigue a la carga. Flag 2: un viaje al azar entre la carga y la siguiente carga de la cadena.
    Una carga marcada tras el último viaje no tiene viaje siguiente."""
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
    """Fase 2. Elimina las cargas marcadas y quita los viajes que indica el paper (`_viajes_a_quitar`). Decisión (d): si
    sin esas cargas la cadena se queda sin batería, quita también los viajes siguientes (o, si no quedan, los últimos)
    hasta que sea factible. Devuelve la solución (sin costo calculado) y los viajes quitados, para la post-reparación.
    No modifica `sol`; las cadenas sin cargas marcadas se comparten."""
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
    """Ajuste de cargadores y potencia completo (Fig. 5): ajuste fino y, si quedaron cargas marcadas, quitar viajes.
    No modifica `sol` (comparte con ella las cadenas que no cambian). La solución que devuelve cumple los límites de todos
    los depósitos."""
    return remove_flagged_trips(fine_tune(sol, inst, rng), inst, rng)
