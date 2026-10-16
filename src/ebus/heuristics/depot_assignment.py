"""Reasignación de depósitos. No es específico del ALNS: cualquier método (baseline, MILP,
ALNS) necesita decidir el depósito de inicio y de retorno de cada bus, así que ambas
funciones son reutilizables por todos (ver `src/ebus/baseline/greedy_construction.py`)."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack

from ebus.heuristics.soc_adjustment import es_factible, recompute_soc
from ebus.model.cost import soc_at_return
from ebus.model.entities import Instance, Solution, TripChain, recharge_events

_RETORNO_INFACTIBLE = 1e6


def assign_start_depots(sol: Solution, inst: Instance) -> None:
    """Cada bus sale del depósito más cercano a la estación de inicio de su primer viaje."""
    for chain in sol.chains:
        if not chain.trips:
            continue
        chain.depot_id = inst.nearest_depot(chain.trips[0].start_station).id


def _ultimo_lugar(chain: TripChain) -> tuple[str, int]:
    """Desde dónde y a qué hora sale el bus hacia el depósito de retorno: el fin de su último viaje o, si carga después
    de él (`recharges[len(trips)]`), el depósito y el fin de esa carga."""
    lugar, t = chain.trips[-1].end_station, chain.trips[-1].end_time
    for carga in recharge_events(chain, len(chain.trips)):
        lugar, t = carga.depot_id, carga.end_time
    return lugar, t


def retorno_factible(chain: TripChain, depot_id: str, inst: Instance) -> bool:
    """El bus puede volver a `depot_id`: llega antes del fin del horizonte y con SOC ≥ `soc_min`."""
    lugar, t = _ultimo_lugar(chain)
    if t + inst.deadhead(lugar, depot_id) > inst.horizon_end:
        return False
    return soc_at_return(replace(chain, return_depot_id=depot_id), inst) >= inst.soc_min


def assign_return_depots(sol: Solution, inst: Instance) -> None:
    """Asigna el retorno de cada bus minimizando el tiempo de deadhead total, exigiendo que el número de buses que vuelve
    a cada depósito coincida con el número que salió de él (restricción (6)). Los buses son intercambiables (flota
    homogénea): lo único que importa es el conteo por depósito, no qué bus físico vuelve a cuál.

    Es el algoritmo húngaro del paper, resuelto como problema de transporte (buses × depósitos, con el balance como
    restricción) y **solo entre retornos factibles** (`retorno_factible`): con varios depósitos, mandar un bus a uno
    lejano puede dejarlo sin batería o fuera del horizonte (docs/decisiones/instancias.md, "Dos terminales"). Con un solo depósito todos
    vuelven a él.

    Si con los depósitos de salida actuales el balance obliga a algún retorno infactible (p. ej., dos buses que solo
    alcanzan a volver a un depósito del que salió uno), ese bus pasa a **salir** de un depósito al que sí puede volver,
    si su cadena sigue siendo factible desde ahí (`_mover_salidas`), y se reparte de nuevo. Si no hay forma, el retorno
    infactible queda y el validador lo reporta."""
    for _ in range(len(sol.chains) + 1):
        _repartir_retornos(sol, inst)
        if not _mover_salidas(sol, inst):
            return


def _mover_salidas(sol: Solution, inst: Instance) -> bool:
    """A cada bus con un retorno infactible le cambia el depósito de salida por uno al que sí puede volver (el más
    cercano a su último lugar primero), recalculando su SOC, si la cadena queda factible saliendo y volviendo ahí.
    Devuelve si cambió alguno. Usa `recompute_soc` y `es_factible`, que recorren la cadena completa."""
    cambio = False
    for i, chain in enumerate(sol.chains):
        if not chain.trips or retorno_factible(chain, chain.return_depot_id, inst):
            continue
        lugar, _ = _ultimo_lugar(chain)
        opciones = sorted((d.id for d in inst.depots if d.id != chain.depot_id and retorno_factible(chain, d.id, inst)),
                          key=lambda d: (inst.deadhead(lugar, d), d))
        for depot_id in opciones:
            nueva = chain.copy()
            nueva.depot_id = nueva.return_depot_id = depot_id
            recompute_soc(nueva, inst)
            if es_factible(nueva, inst):
                sol.chains[i] = nueva
                cambio = True
                break
    return cambio


def _repartir_retornos(sol: Solution, inst: Instance) -> None:
    """Problema de transporte de `assign_return_depots` con los depósitos de salida actuales."""
    chains = [c for c in sol.chains if c.trips]
    if not chains:
        return
    salidas = Counter(c.depot_id for c in chains)
    depositos = sorted(salidas)
    if len(depositos) == 1:
        for chain in chains:
            chain.return_depot_id = depositos[0]
        return

    n, m = len(chains), len(depositos)
    costo = np.empty((n, m))
    for i, chain in enumerate(chains):
        lugar, _ = _ultimo_lugar(chain)
        for j, depot_id in enumerate(depositos):
            costo[i, j] = inst.deadhead(lugar, depot_id)
            if not retorno_factible(chain, depot_id, inst):
                costo[i, j] += _RETORNO_INFACTIBLE

    columnas = np.arange(n * m)
    por_bus = coo_matrix((np.ones(n * m), (np.repeat(np.arange(n), m), columnas)), shape=(n, n * m))
    por_deposito = coo_matrix((np.ones(n * m), (np.tile(np.arange(m), n), columnas)), shape=(m, n * m))
    lado_derecho = np.concatenate([np.ones(n), [salidas[d] for d in depositos]])
    res = linprog(costo.ravel(), A_eq=vstack([por_bus, por_deposito]), b_eq=lado_derecho, bounds=(0, 1),
                  method="highs-ds")
    if res.status != 0:
        raise RuntimeError(f"No se pudo asignar los depósitos de retorno: {res.message}")
    x = res.x.reshape(n, m)
    if not np.allclose(x, np.round(x), atol=1e-6):
        raise RuntimeError("La asignación de retornos no salió entera")
    for i, chain in enumerate(chains):
        chain.return_depot_id = depositos[int(np.argmax(x[i]))]
