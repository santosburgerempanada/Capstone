from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np

from ebus.heuristics.soc_adjustment import adjust_for_insertion
from ebus.model.cost import chain_variable_cost
from ebus.model.entities import Instance, Solution, Trip, TripChain


@dataclass
class Insertion:
    trip: Trip
    chain_idx: int
    position: int
    cost: float
    new_chain: TripChain


def _inserciones_en_cadena(trip: Trip, chain: TripChain, chain_idx: int, costo_base: float,
                           inst: Instance) -> list[Insertion]:
    opciones = []
    for pos in range(len(chain.trips) + 1):
        nueva = adjust_for_insertion(chain, pos, trip, inst)
        if nueva is not None:
            opciones.append(Insertion(trip, chain_idx, pos, chain_variable_cost(nueva, inst) - costo_base, nueva))
    return opciones


def feasible_insertions(trip: Trip, sol: Solution, inst: Instance) -> list[Insertion]:
    return [ins for j, chain in enumerate(sol.chains)
            for ins in _inserciones_en_cadena(trip, chain, j, chain_variable_cost(chain, inst), inst)]


def _clave(ins: Insertion) -> tuple:
    return ins.cost, ins.trip.id, ins.chain_idx, ins.position


class _Opciones:

    def __init__(self, sol: Solution, pendientes: list[Trip], inst: Instance):
        self.sol, self.inst = sol, inst
        self.pendientes = {t.id: t for t in sorted(pendientes, key=lambda t: t.id)}
        self.costos = [chain_variable_cost(c, inst) for c in sol.chains]
        self.por_viaje: dict[str, dict[int, list[Insertion]]] = {
            tid: {j: _inserciones_en_cadena(t, c, j, self.costos[j], inst) for j, c in enumerate(sol.chains)}
            for tid, t in self.pendientes.items()
        }

    def de(self, trip_id: str) -> list[Insertion]:
        return sorted((ins for lista in self.por_viaje[trip_id].values() for ins in lista), key=_clave)

    def aplicar(self, ins: Insertion) -> None:
        j = ins.chain_idx
        self.sol.chains[j] = ins.new_chain
        self.costos[j] = chain_variable_cost(ins.new_chain, self.inst)
        del self.pendientes[ins.trip.id]
        del self.por_viaje[ins.trip.id]
        for tid, t in self.pendientes.items():
            self.por_viaje[tid][j] = _inserciones_en_cadena(t, ins.new_chain, j, self.costos[j], self.inst)


def greedy_repair(
    sol: Solution, removed: list[Trip], inst: Instance, rng: np.random.Generator | None = None
) -> tuple[Solution, list[Trip]]:
    reparada = sol.copy()
    reparada.cost = None
    opciones = _Opciones(reparada, removed, inst)
    while opciones.pendientes:
        candidatas = [ins for tid in opciones.pendientes for ins in opciones.de(tid)[:1]]
        if not candidatas:
            break
        opciones.aplicar(min(candidatas, key=_clave))
    return reparada, list(opciones.pendientes.values())


def regret_repair(
    sol: Solution, removed: list[Trip], inst: Instance, rng: np.random.Generator | None = None, p: int = 2
) -> tuple[Solution, list[Trip]]:
    reparada = sol.copy()
    reparada.cost = None
    opciones = _Opciones(reparada, removed, inst)
    while opciones.pendientes:
        mejor: tuple | None = None
        for tid in opciones.pendientes:
            lista = opciones.de(tid)
            if not lista:
                continue
            regret = float("inf") if len(lista) < p else sum(o.cost - lista[0].cost for o in lista[1:p])
            clave = (-regret, lista[0].cost, tid)
            if mejor is None or clave < mejor[0]:
                mejor = (clave, lista[0])
        if mejor is None:
            break
        opciones.aplicar(mejor[1])
    return reparada, list(opciones.pendientes.values())


def greedy_noise_repair(
    sol: Solution, removed: list[Trip], inst: Instance, rng: np.random.Generator, noise: float = 0.15
) -> tuple[Solution, list[Trip]]:
    reparada = sol.copy()
    reparada.cost = None
    opciones = _Opciones(reparada, removed, inst)
    while opciones.pendientes:
        mejor: tuple[float, Insertion] | None = None
        for tid in opciones.pendientes:
            for ins in opciones.de(tid):
                clave = ins.cost + abs(ins.cost) * rng.uniform(-noise, noise)
                if mejor is None or clave < mejor[0]:
                    mejor = (clave, ins)
        if mejor is None:
            break
        opciones.aplicar(mejor[1])
    return reparada, list(opciones.pendientes.values())


RepairOperator = Callable[[Solution, list[Trip], Instance, np.random.Generator], tuple[Solution, list[Trip]]]


def operadores_repair(alns_cfg: dict, extensiones: bool = False) -> dict[str, RepairOperator]:
    todos = {
        "greedy": lambda: greedy_repair,
        "regret": lambda: partial(regret_repair, p=alns_cfg["regret_p"]),
        "greedy_noise": lambda: partial(greedy_noise_repair, **alns_cfg["greedy_noise_repair"]),
    }
    nombres = list(alns_cfg.get("operadores_repair") or ("greedy", "regret"))
    if extensiones and "greedy_noise" not in nombres:
        nombres.append("greedy_noise")
    desconocidos = [n for n in nombres if n not in todos]
    if desconocidos:
        raise ValueError(f"Operadores repair desconocidos: {desconocidos}; los posibles son {list(todos)}")
    return {n: todos[n]() for n in nombres}
