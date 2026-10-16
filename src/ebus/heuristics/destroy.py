"""Operadores destroy (Apéndice A.(1) del paper). Cada uno quita viajes de la solución y devuelve la solución
destruida junto con los viajes quitados, que luego reinserta el repair.

Los tres operadores del paper son `random_destroy`, `worst_destroy` y `similarity_destroy`. En cada iteración quitan un
número de viajes **al azar entre 1 y `d_max`** (decisión del equipo, ver docs/decisiones/alns.md). Todos reciben el `rng`
de la librería `alns` (un `numpy.random.Generator`) y no modifican la solución que reciben: trabajan sobre una copia.

Más abajo quedan, como **extensión opcional desactivada** (no son del paper), los operadores `chain_destroy`,
`string_destroy` y `cluster_destroy` que aportó `Yusolis`. `operadores_destroy` arma el registro de operadores activos
a partir de la config."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from functools import partial

import numpy as np

from ebus.heuristics.soc_adjustment import es_factible, recompute_soc
from ebus.model.entities import Instance, Solution, Trip, TripChain, recharge_events


def quitar_viajes(sol: Solution, viajes: Iterable[Trip], inst: Instance) -> Solution:
    """Copia de `sol` sin los `viajes` (se reconocen por su `id`), lista para que el repair los reinserte.

    - No modifica `sol`: devuelve una solución nueva e independiente.
    - Un bus que se queda sin viajes desaparece. Los demás conservan su depósito, su `bus_id` y sus viajes restantes.
    - Las cargas se reacomodan: `recharges[k]` va entre los viajes `k-1` y `k`, así que al quitar un viaje las cargas de
      los intervalos que lo rodeaban se **concatenan en orden cronológico** en el intervalo resultante (no se pierde
      ninguna). La excepción son las que quedarían antes del primer viaje, porque cargar en el arco inicial está
      prohibido (restricción (13)): esas se descartan.
    - El SOC de los buses modificados se recalcula con `recompute_soc`; los buses sin cambios quedan idénticos.
    - El costo de la solución resultante no está calculado (`cost = None`): aún faltan los viajes quitados."""
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
    """`resto` es una cadena a la que ya se le quitaron viajes (lista vacía si se quedó sin viajes). Si dejó de ser
    factible (por batería o por horarios), le quita el viaje en `posicion` (el que seguía al primer viaje quitado) y los
    siguientes, o los últimos si ya no quedan, hasta que vuelva a serlo o se quede sin viajes. Devuelve la cadena (en
    una lista, vacía si desapareció) y los viajes que hubo que quitar de más."""
    extra: list[Trip] = []
    while resto and not es_factible(resto[0], inst):
        restantes = resto[0].trips
        posicion = min(posicion, len(restantes) - 1)
        extra.append(restantes[posicion])
        resto = quitar_viajes(Solution(chains=resto), [restantes[posicion]], inst).chains
    return resto, extra


def quitar_viajes_factible(sol: Solution, viajes: Iterable[Trip], inst: Instance) -> tuple[Solution, list[Trip]]:
    """`quitar_viajes` que deja factibles todas las cadenas. Lo usan todos los destroy.

    Quitar un viaje puede **romper** una cadena: si hacer el viaje gasta menos energía (o tiempo) que el deadhead directo
    que lo reemplaza, sin él el bus puede quedarse sin batería o no llegar a tiempo. El paper no considera este caso;
    nuestros datos no cumplen la desigualdad triangular entre viajes y deadheads (decisión del equipo, ver
    docs/decisiones/alns.md). En las cadenas que quedan infactibles se quitan también los viajes siguientes
    (`quitar_hasta_factible`, la misma regla del ajuste de cargadores). Devuelve la solución destruida y todos los viajes
    quitados: los pedidos y, al final, los quitados de más."""
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
    """Número de viajes a quitar en esta iteración: uniforme entre 1 y `d_max`, sin pasar de los que hay."""
    return min(int(rng.integers(1, d_max + 1)), disponibles)


def random_destroy(
    sol: Solution, inst: Instance, d_max: int, rng: np.random.Generator
) -> tuple[Solution, list[Trip]]:
    """Destroy aleatorio: quita viajes al azar."""
    viajes = _todos_los_viajes(sol)
    k = _cuantos(rng, d_max, len(viajes))
    quitados = [viajes[i] for i in rng.choice(len(viajes), size=k, replace=False)] if k else []
    return quitar_viajes_factible(sol, quitados, inst)


def worst_destroy(
    sol: Solution, inst: Instance, d_max: int, rng: np.random.Generator, *, l_min: int, t_max: int
) -> tuple[Solution, list[Trip]]:
    """Destroy "peor". Primero vacía cadenas con menos de `l_min` viajes, elegidas al azar entre las cortas y solo si
    caben completas en el número de viajes a quitar; después quita los viajes cuya holgura (`slack`) con el viaje anterior
    **y** con el siguiente supera `t_max` minutos, de la peor (mayor suma) a la mejor. No completa con viajes al azar: si
    no hay más candidatos, quita menos. Los viajes de los extremos de una cadena no son candidatos."""
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
    """(A.3): TR_ab = |t_a1 − t_b1| + |t_a2 − t_b2| + d(s_a1, s_b1) + d(s_a2, s_b2), con t1 y t2 las horas de inicio y
    fin, y `d` el tiempo de deadhead entre las estaciones de inicio y las de fin."""
    return (
        abs(a.start_time - b.start_time)
        + abs(a.end_time - b.end_time)
        + inst.deadhead(a.start_station, b.start_station)
        + inst.deadhead(a.end_station, b.end_station)
    )


def similarity_destroy(
    sol: Solution, inst: Instance, d_max: int, rng: np.random.Generator, *, power_factor: float
) -> tuple[Solution, list[Trip]]:
    """Destroy por similitud temporal (A.3). Parte de un viaje al azar; en cada paso toma como referencia un viaje al azar
    del conjunto ya quitado, ordena los viajes restantes por similitud (menor `TR` primero) y quita el que está en el
    percentil `r^p` de esa lista, con `r` uniforme en (0, 1) y `p = power_factor`. Los restantes se reordenan en cada
    paso, porque el conjunto de viajes disponibles cambia."""
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
    """Route removal: elimina buses completos con baja utilización (cada bus eliminado ahorra su
    costo fijo). Prefiere cadenas cortas (índice ~ r^bias). Puede superar d_max hasta
    budget_factor * d_max para poder vaciar una cadena entera."""
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
    """String removal (SISR): desde un viaje semilla, recorre los viajes más cercanos (hora de inicio
    + deadhead entre orígenes) y de cada cadena distinta quita un tramo CONSECUTIVO de hasta l_max
    viajes que contiene al viaje visitado."""
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
    """Identificador de línea del viaje (usa `line` o, si no existe, `route_id`)."""
    return getattr(t, "line", getattr(t, "route_id", None))


def cluster_destroy(
    sol: Solution, inst: Instance, d_max: int, rng: np.random.Generator, w_station: float = 20.0, w_line: float = 10.0
) -> tuple[Solution, list[Trip]]:
    """Cluster espacio-temporal: quita los d_max viajes más relacionados con una semilla aleatoria.
    Relación = |Δ hora de inicio| + w_station·[otro terminal de origen] + w_line·[otra línea],
    con ruido U(0.8, 1.2). Los pesos están en las unidades de start_time (se asumen minutos)."""
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
    """Vacía un bus completo, de cualquier largo (extensión, no es del paper; docs/decisiones/alns.md): ordena los buses
    por número de viajes (menos primero; a igual número, por `bus_id`) y elige el del percentil `r^bias`, con sesgo a los
    de menos viajes pero pudiendo elegir cualquiera. Quita **todos** sus viajes, aunque sean más que `d_max`, para que el
    repair los reparta entre los demás buses y el bus desaparezca."""
    cadenas = sorted((c for c in sol.chains if c.trips), key=lambda c: (len(c.trips), c.bus_id or ""))
    if not cadenas:
        return quitar_viajes_factible(sol, [], inst)
    elegida = cadenas[int(rng.random() ** bias * len(cadenas))]
    return quitar_viajes_factible(sol, list(elegida.trips), inst)


_DESTROY_PAPER = ("random", "worst", "similarity")
_DESTROY_EXTENSIONES = ("chain", "string", "cluster", "vaciar_bus")


def operadores_destroy(alns_cfg: dict, extensiones: bool = False) -> dict[str, DestroyOperator]:
    """Operadores destroy activos, con sus parámetros tomados de la sección `alns` de la config. Los activos son los de
    `alns.operadores_destroy` (por nombre; si no está, los tres del paper); `extensiones=True` activa además todas las
    extensiones (`chain`, `string`, `cluster` y `vaciar_bus`, que no son del paper)."""
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
