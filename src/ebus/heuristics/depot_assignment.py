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
    for chain in sol.chains:
        if not chain.trips:
            continue
        chain.depot_id = inst.nearest_depot(chain.trips[0].start_station).id


def _ultimo_lugar(chain: TripChain) -> tuple[str, int]:
    lugar, t = chain.trips[-1].end_station, chain.trips[-1].end_time
    for carga in recharge_events(chain, len(chain.trips)):
        lugar, t = carga.depot_id, carga.end_time
    return lugar, t


def retorno_factible(chain: TripChain, depot_id: str, inst: Instance) -> bool:
    lugar, t = _ultimo_lugar(chain)
    if t + inst.deadhead(lugar, depot_id) > inst.horizon_end:
        return False
    return soc_at_return(replace(chain, return_depot_id=depot_id), inst) >= inst.soc_min


def assign_return_depots(sol: Solution, inst: Instance) -> None:
    for _ in range(len(sol.chains) + 1):
        _repartir_retornos(sol, inst)
        if not _mover_salidas(sol, inst):
            return


def _mover_salidas(sol: Solution, inst: Instance) -> bool:
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
