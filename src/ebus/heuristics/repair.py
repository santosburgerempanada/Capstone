"""Operadores repair (Apéndice A.(2) del paper). Reinsertan los viajes quitados por el destroy, de a uno, en la mejor
posición según el operador. Cada inserción usa la sub-rutina de ajuste de SOC (`adjust_for_insertion`).

Decisiones del equipo (ver docs/decisiones/alns.md):
- **Costo de una inserción** = cambio en el costo variable del bus (`chain_variable_cost`: deadhead por km, carga diurna y
  carga nocturna), no solo el deadhead extra en minutos de la ec. (20) del paper. Así una posición que obliga a cargar de
  día cuesta más que una que no.
- **Solo se inserta en buses existentes**, como en el paper. Los viajes sin posición factible se devuelven para la
  post-reparación, que es la que abre buses nuevos.
- **Regret-p:** un viaje con menos de `p` posiciones factibles tiene prioridad máxima (puede quedarse sin lugar).
- Los límites de cargadores y potencia compartidos están relajados, como en el paper (los corrige el ajuste de cargadores).
- Desempates fijos (costo, id del viaje, índice del bus, posición): la misma entrada da el mismo resultado.

`greedy_noise_repair` (de `Yusolis`, no es del paper) queda como extensión opcional desactivada."""

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
    """Todas las posiciones factibles de `trip` en un bus, con su costo."""
    opciones = []
    for pos in range(len(chain.trips) + 1):
        nueva = adjust_for_insertion(chain, pos, trip, inst)
        if nueva is not None:
            opciones.append(Insertion(trip, chain_idx, pos, chain_variable_cost(nueva, inst) - costo_base, nueva))
    return opciones


def feasible_insertions(trip: Trip, sol: Solution, inst: Instance) -> list[Insertion]:
    """Todas las inserciones factibles de `trip` en los buses de `sol` (en tiempo y SOC, vía `adjust_for_insertion`), con
    su costo. No modifica `sol`."""
    return [ins for j, chain in enumerate(sol.chains)
            for ins in _inserciones_en_cadena(trip, chain, j, chain_variable_cost(chain, inst), inst)]


def _clave(ins: Insertion) -> tuple:
    return ins.cost, ins.trip.id, ins.chain_idx, ins.position


class _Opciones:
    """Opciones de inserción de los viajes pendientes, por bus. Tras insertar en un bus, solo se recalculan las opciones
    de ese bus: las de los demás siguen valiendo."""

    def __init__(self, sol: Solution, pendientes: list[Trip], inst: Instance):
        self.sol, self.inst = sol, inst
        self.pendientes = {t.id: t for t in sorted(pendientes, key=lambda t: t.id)}
        self.costos = [chain_variable_cost(c, inst) for c in sol.chains]
        self.por_viaje: dict[str, dict[int, list[Insertion]]] = {
            tid: {j: _inserciones_en_cadena(t, c, j, self.costos[j], inst) for j, c in enumerate(sol.chains)}
            for tid, t in self.pendientes.items()
        }

    def de(self, trip_id: str) -> list[Insertion]:
        """Opciones del viaje, de la más barata a la más cara."""
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
    """(A.4): en cada paso inserta el par (viaje, posición) de menor costo, entre todos los viajes pendientes. Devuelve la
    solución reparada (una copia) y los viajes que no tuvieron ninguna posición factible. `rng` no se usa (el operador es
    determinista); se recibe porque la librería `alns` se lo entrega a todos los repair."""
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
    """(A.5): inserta primero el viaje con mayor arrepentimiento `Σ_{j≤p} (f_ij − f_i1)`, la diferencia entre su mejor
    posición y sus `p − 1` siguientes, en su mejor posición. Un viaje con menos de `p` posiciones factibles tiene prioridad
    máxima; entre iguales, va primero el de inserción más barata. Devuelve la solución reparada (una copia) y los viajes que
    no tuvieron ninguna posición factible. `rng` no se usa (el operador es determinista)."""
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
    """Greedy con ruido: en cada paso inserta el par (viaje, posición) de menor costo PERTURBADO,
    cost + |cost|·U(-noise, noise). El ruido solo orienta la elección; el costo real de la
    inserción (ins.cost / new_chain) no se altera. Sirve para diversificar frente a greedy_repair.
    Devuelve los viajes sin inserción factible (los maneja el post-repair)."""
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
    """Operadores repair activos, con sus parámetros de la sección `alns` de la config (`regret_p`,
    `greedy_noise_repair.noise`). Los activos son los de `alns.operadores_repair` (por nombre; si no está, los dos del
    paper); `extensiones=True` agrega `greedy_noise` (no es del paper)."""
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
