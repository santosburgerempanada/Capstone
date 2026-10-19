from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from functools import partial

import numpy as np

from ebus.heuristics.soc_adjustment import es_factible, recompute_soc
from ebus.model.entities import Instance, Solution, Trip, TripChain, recharge_events


def quitar_viajes(sol: Solution, viajes: Iterable[Trip], inst: Instance) -> Solution:
    ids = {t.id for t in viajes}
    cadenas = []
    for cadena in sol.chains:
        conservados = [i for i, t in enumerate(cadena.trips) if t.id not in ids]
        if len(conservados) == len(cadena.trips):
            cadenas.append(cadena.copy())
            continue
        if not conservados:
            continue
        nueva = cadena.copy()
        nueva.trips = [cadena.trips[i] for i in conservados]
        recargas: dict[int, list] = {}
        previo = -1
        for j, i in enumerate(conservados + [len(cadena.trips)]):
            claves = [0] if j == 0 else range(previo + 1, i + 1)
            eventos = [replace(e) for k in claves for e in recharge_events(cadena, k)]
            if eventos:
                recargas[j] = eventos
            previo = i
        nueva.recharges = recargas
        recompute_soc(nueva, inst)
        cadenas.append(nueva)
    return Solution(chains=cadenas, unassigned=list(sol.unassigned), cost=None)


def quitar_hasta_factible(resto: list[TripChain], posicion: int, inst: Instance) -> tuple[list[TripChain], list[Trip]]:
    extra: list[Trip] = []
    while resto and not es_factible(resto[0], inst):
        restantes = resto[0].trips
        posicion = min(posicion, len(restantes) - 1)
        extra.append(restantes[posicion])
        resto = quitar_viajes(Solution(chains=resto), [restantes[posicion]], inst).chains
    return resto, extra


def quitar_viajes_factible(sol: Solution, viajes: Iterable[Trip], inst: Instance) -> tuple[Solution, list[Trip]]:
    viajes = list(viajes)
    ids = {t.id for t in viajes}
    cadenas: list[TripChain] = []
    extra: list[Trip] = []
    for cadena in sol.chains:
        posiciones = [i for i, t in enumerate(cadena.trips) if t.id in ids]
        if not posiciones:
            cadenas.append(cadena.copy())
            continue
        resto = quitar_viajes(Solution(chains=[cadena]), viajes, inst).chains
        resto, mas = quitar_hasta_factible(resto, posiciones[0], inst)
        cadenas += resto
        extra += mas
    return Solution(chains=cadenas, unassigned=list(sol.unassigned), cost=None), viajes + extra


def _todos_los_viajes(sol: Solution) -> list[Trip]:
    return [t for c in sol.chains for t in c.trips]


def _cuantos(rng: np.random.Generator, d_max: int, disponibles: int) -> int:
    return min(int(rng.integers(1, d_max + 1)), disponibles)


def random_destroy(
    sol: Solution, inst: Instance, d_max: int, rng: np.random.Generator
) -> tuple[Solution, list[Trip]]:
    viajes = _todos_los_viajes(sol)
    k = _cuantos(rng, d_max, len(viajes))
    quitados = [viajes[i] for i in rng.choice(len(viajes), size=k, replace=False)] if k else []
    return quitar_viajes_factible(sol, quitados, inst)


def worst_destroy(
    sol: Solution, inst: Instance, d_max: int, rng: np.random.Generator, *, l_min: int, t_max: int
) -> tuple[Solution, list[Trip]]:
    k = _cuantos(rng, d_max, sum(len(c.trips) for c in sol.chains))
    quitados: list[Trip] = []

    cortas = [c for c in sol.chains if len(c.trips) < l_min]
    for i in rng.permutation(len(cortas)):
        if len(quitados) + len(cortas[i].trips) <= k:
            quitados += cortas[i].trips

    ids = {t.id for t in quitados}
    candidatos = []
    for cadena in sol.chains:
        for pos in range(1, len(cadena.trips) - 1):
            antes = inst.slack(cadena.trips[pos - 1], cadena.trips[pos])
            despues = inst.slack(cadena.trips[pos], cadena.trips[pos + 1])
            if antes > t_max and despues > t_max and cadena.trips[pos].id not in ids:
                candidatos.append((antes + despues, cadena.trips[pos]))
    for _, viaje in sorted(candidatos, key=lambda c: (-c[0], c[1].id)):
        if len(quitados) >= k:
            break
        quitados.append(viaje)
    return quitar_viajes_factible(sol, quitados, inst)


def time_similarity(a: Trip, b: Trip, inst: Instance) -> float:
    return (
        abs(a.start_time - b.start_time)
        + abs(a.end_time - b.end_time)
        + inst.deadhead(a.start_station, b.start_station)
        + inst.deadhead(a.end_station, b.end_station)
    )


def similarity_destroy(
    sol: Solution, inst: Instance, d_max: int, rng: np.random.Generator, *, power_factor: float
) -> tuple[Solution, list[Trip]]:
    viajes = _todos_los_viajes(sol)
    k = _cuantos(rng, d_max, len(viajes))
    quitados: list[Trip] = []
    if k:
        quitados.append(viajes[int(rng.integers(len(viajes)))])
    while len(quitados) < k:
        referencia = quitados[int(rng.integers(len(quitados)))]
        ids = {t.id for t in quitados}
        restantes = sorted(
            (t for t in viajes if t.id not in ids), key=lambda t: (time_similarity(referencia, t, inst), t.id)
        )
        quitados.append(restantes[int(rng.random() ** power_factor * len(restantes))])
    return quitar_viajes_factible(sol, quitados, inst)


def _viajes_al_azar(sol: Solution, d_max: int, rng: np.random.Generator) -> list[Trip]:
    viajes = _todos_los_viajes(sol)
    return [viajes[i] for i in rng.choice(len(viajes), size=min(d_max, len(viajes)), replace=False)]


def chain_destroy(
    sol: Solution, inst: Instance, d_max: int, rng: np.random.Generator, budget_factor: int = 2, bias: float = 3
) -> tuple[Solution, list[Trip]]:
    budget = budget_factor * d_max
    pool = sorted((c for c in sol.chains if len(c.trips) <= budget), key=lambda c: len(c.trips))
    removed: list[Trip] = []
    while pool and len(removed) < d_max:
        c = pool.pop(int(rng.random() ** bias * len(pool)))
        if len(removed) + len(c.trips) <= budget:
            removed += c.trips
    if not removed:
        removed = _viajes_al_azar(sol, d_max, rng)
    return quitar_viajes_factible(sol, removed, inst)


def string_destroy(
    sol: Solution, inst: Instance, d_max: int, rng: np.random.Generator, l_max: int = 4
) -> tuple[Solution, list[Trip]]:
    trips = _todos_los_viajes(sol)
    chain_of = {id(t): c for c in sol.chains for t in c.trips}
    seed = trips[int(rng.integers(len(trips)))]
    near = sorted(
        trips,
        key=lambda t: abs(t.start_time - seed.start_time) + inst.deadhead(t.start_station, seed.start_station),
    )
    removed: list[Trip] = []
    used: set[int] = set()
    for t in near:
        if len(removed) >= d_max:
            break
        c = chain_of[id(t)]
        if id(c) in used:
            continue
        used.add(id(c))
        length = min(len(c.trips), int(rng.integers(1, l_max + 1)), d_max - len(removed))
        pos = next(i for i, x in enumerate(c.trips) if x is t)
        start = max(0, min(pos - int(rng.integers(0, length)), len(c.trips) - length))
        removed += c.trips[start:start + length]
    return quitar_viajes_factible(sol, removed, inst)


def _line(t: Trip):
    return getattr(t, "line", getattr(t, "route_id", None))


def cluster_destroy(
    sol: Solution, inst: Instance, d_max: int, rng: np.random.Generator, w_station: float = 20.0, w_line: float = 10.0
) -> tuple[Solution, list[Trip]]:
    trips = _todos_los_viajes(sol)
    seed = trips[int(rng.integers(len(trips)))]

    def relatedness(t: Trip) -> float:
        base = (
            abs(t.start_time - seed.start_time)
            + w_station * (t.start_station != seed.start_station)
            + w_line * (_line(t) != _line(seed))
        )
        return base * rng.uniform(0.8, 1.2)

    removed = sorted(trips, key=relatedness)[:min(d_max, len(trips))]
    return quitar_viajes_factible(sol, removed, inst)


DestroyOperator = Callable[[Solution, Instance, int, np.random.Generator], tuple[Solution, list[Trip]]]


def vaciar_bus_destroy(
    sol: Solution, inst: Instance, d_max: int, rng: np.random.Generator, bias: float = 3
) -> tuple[Solution, list[Trip]]:
    cadenas = sorted((c for c in sol.chains if c.trips), key=lambda c: (len(c.trips), c.bus_id or ""))
    if not cadenas:
        return quitar_viajes_factible(sol, [], inst)
    elegida = cadenas[int(rng.random() ** bias * len(cadenas))]
    return quitar_viajes_factible(sol, list(elegida.trips), inst)


_DESTROY_PAPER = ("random", "worst", "similarity")
_DESTROY_EXTENSIONES = ("chain", "string", "cluster", "vaciar_bus")


def operadores_destroy(alns_cfg: dict, extensiones: bool = False) -> dict[str, DestroyOperator]:
    worst, similarity = alns_cfg["worst_destroy"], alns_cfg["similarity_destroy"]
    todos: dict[str, Callable[[], DestroyOperator]] = {
        "random": lambda: random_destroy,
        "worst": lambda: partial(worst_destroy, l_min=worst["L_min"], t_max=worst["T_max_min"]),
        "similarity": lambda: partial(similarity_destroy, power_factor=similarity["power_factor"]),
        "chain": lambda: partial(chain_destroy, **alns_cfg["chain_destroy"]),
        "string": lambda: partial(string_destroy, **alns_cfg["string_destroy"]),
        "cluster": lambda: partial(cluster_destroy, **alns_cfg["cluster_destroy"]),
        "vaciar_bus": lambda: partial(vaciar_bus_destroy, **alns_cfg["vaciar_bus"]),
    }
    nombres = list(alns_cfg.get("operadores_destroy") or _DESTROY_PAPER)
    if extensiones:
        nombres += [n for n in _DESTROY_EXTENSIONES if n not in nombres]
    desconocidos = [n for n in nombres if n not in todos]
    if desconocidos:
        raise ValueError(f"Operadores destroy desconocidos: {desconocidos}; los posibles son {list(todos)}")
    return {n: todos[n]() for n in nombres}
