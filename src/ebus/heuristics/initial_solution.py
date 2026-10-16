"""Solución inicial factible (Sección 3.1, Fig. 4): pre-conexión de viajes + modelo (17)-(19).
Plan: docs/planes/plan_solucion_inicial.md; decisiones: docs/decisiones/alns.md (entradas del 2026-10-11)."""

from __future__ import annotations

import bisect
import logging
import time
from collections import defaultdict
from collections.abc import Callable

import numpy as np

from ebus.heuristics.capacidad import CapacidadLibre
from ebus.model.entities import Instance, Solution, Trip, TripChain

registro = logging.getLogger(__name__)


def build_trip_pairs(trips: list[Trip], T_tc: int) -> list[tuple[Trip, Trip]]:
    """Pares pre-conectados (Sección 3.1): (a, b) si `a` termina en la estación donde parte `b` y la espera entre ambos
    está entre 0 y `T_tc` minutos (`0 <= b.start_time - a.end_time <= T_tc`). Se enumeran todos: un viaje puede estar en
    varios pares. Como la estación es la misma, no hay deadhead entre `a` y `b`.

    Orden determinista: por `a` (salida, id) y, para cada `a`, por `b` (salida, id)."""
    if T_tc < 0:
        raise ValueError("T_tc debe ser no negativo")
    por_estacion: dict[str, list[Trip]] = defaultdict(list)
    for b in trips:
        por_estacion[b.start_station].append(b)
    salidas: dict[str, list[int]] = {}
    for estacion, viajes in por_estacion.items():
        viajes.sort(key=lambda t: (t.start_time, t.id))
        salidas[estacion] = [t.start_time for t in viajes]

    pares: list[tuple[Trip, Trip]] = []
    for a in sorted(trips, key=lambda t: (t.start_time, t.id)):
        candidatos = por_estacion.get(a.end_station, [])
        tiempos = salidas.get(a.end_station, [])
        desde = bisect.bisect_left(tiempos, a.end_time)
        hasta = bisect.bisect_right(tiempos, a.end_time + T_tc)
        pares.extend((a, b) for b in candidatos[desde:hasta] if b.id != a.id)
    return pares


def sample_sub_chain(
    pairs: list[tuple[Trip, Trip]], pool: set[str], N_p: int, inst: Instance, rng: np.random.Generator,
    enlaza: Callable[[Trip, Trip], bool] | None = None,
) -> list[tuple[Trip, ...]]:
    """Sub-cadena Tps_n (Sección 3.1, Fig. 4; decisiones B, C y K de docs/decisiones/alns.md).

    Solo cuentan los pares con sus dos viajes en el pool (`pool`: ids de los viajes sin asignar). Se parte de un par al
    azar y se agregan, también al azar, pares cuyo primer viaje es alcanzable después del último viaje de la sub-cadena
    (`Instance.can_connect`, que incluye el deadhead) y que no repiten viajes, hasta tener `N_p` pares o hasta que no
    quede ninguno válido. Si ningún par tiene sus dos viajes en el pool, la semilla es un viaje suelto del pool, al azar.

    Devuelve los bloques en orden: tuplas `(a, b)` para los pares o `(viaje,)` para la semilla suelta, de modo que se
    pueda quitar el último bloque si la sub-cadena no cabe en un bus (decisión J). `rng` es un `numpy.random.Generator`.
    `enlaza(a, b)` reemplaza a `Instance.can_connect` para decidir si un par sigue a otro: con la red recortada de la
    solución inicial, solo cuentan los enlaces que existen en esa red (`modelo_un_bus.enlaces_de_la_red`)."""
    if N_p < 1:
        raise ValueError("N_p debe ser al menos 1")
    if not pool:
        raise ValueError("El pool de viajes está vacío")
    validos = [(a, b) for a, b in pairs if a.id in pool and b.id in pool]
    if not validos:
        por_id = {t.id: t for t in inst.trips}
        ids = sorted(pool)
        return [(por_id[ids[rng.integers(len(ids))]],)]

    enlaza = enlaza or inst.can_connect
    bloques: list[tuple[Trip, ...]] = [validos[rng.integers(len(validos))]]
    usados = {t.id for t in bloques[0]}
    while len(bloques) < N_p:
        ultimo = bloques[-1][-1]
        candidatos = [(a, b) for a, b in validos
                      if a.id not in usados and b.id not in usados and enlaza(ultimo, a)]
        if not candidatos:
            break
        par = candidatos[rng.integers(len(candidatos))]
        bloques.append(par)
        usados.update(t.id for t in par)
    return bloques


def extend_sub_chain(sub_chain: list[Trip], pool: list[Trip], inst: Instance, time_limit_s: float | None,
                     libre: CapacidadLibre | None = None, k_sucesores: int | None = None,
                     red_completa: bool = False) -> TripChain | None:
    """Modelo (17)-(19): la cadena de un bus que hace todos los viajes de `sub_chain` y el máximo de viajes del `pool`
    (que incluye a los de la sub-cadena); a igual número, el menor deadhead. Con (3)-(6) y (9)-(16) y carga parcial; con
    `libre`, solo carga donde quedan cargadores y potencia libres (decisión E). Devuelve `None` si la sub-cadena no cabe
    en un bus o Gurobi no encuentra solución en `time_limit_s` (decisión J). La red es la de un bus (una pista por
    depósito; conexiones directas a los `k_sucesores` primeros sucesores), o la del MILP con `red_completa=True`
    (decisión G). Arma el modelo para un solo bus; `initial_solution` lo arma una vez para todos. Necesita Gurobi; ver
    `heuristics/modelo_un_bus.py`."""
    from ebus.heuristics.modelo_un_bus import cadena_de_un_bus

    return cadena_de_un_bus(sub_chain, pool, inst, time_limit_s, libre, k_sucesores, red_completa)


def initial_solution(inst: Instance, T_tc: int, N_p: int, rng: np.random.Generator, time_limit_s: float | None,
                     post_repair_time_limit_s: float, use_preconnection: bool = True,
                     k_sucesores: int | None = None) -> Solution:
    """Solución inicial del paper (Sección 3.1, Fig. 2): mientras queden viajes sin asignar, elegir una sub-cadena,
    resolver el modelo (17)-(19) para un bus con todo el pool y quitar del pool los viajes de ese bus. El número de buses
    es el número de vueltas (no se fija antes). Cada bus parte con `inst.soc_initial`.

    - **Un modelo para todos los buses (decisión G):** el modelo de un bus se arma una vez, sobre la red de un bus con los
      `k_sucesores` primeros sucesores de cada viaje más las conexiones de los pares pre-conectados; para cada bus se
      cambian la cobertura (obligatorios y viajes ya asignados) y la capacidad libre. Las sub-cadenas solo encadenan
      pares que están enlazados en esa red, para que siempre quepan en ella.
    - **Sub-cadena que no cabe en un bus (decisión J):** se quita su último par y se reintenta; si ni un par cabe, se usa
      el primer viaje del par solo. Un viaje que ni solo cabe queda pendiente.
    - **Cargadores y potencia (decisión E, cambiada el 2026-10-11):** cada bus solo carga donde quedan cargadores y
      potencia libres después de los buses ya construidos (`CapacidadLibre`).
    - **Al final (decisión I):** el ajuste de cargadores y potencia (`enforce_charging_resources`), como red de
      seguridad, y el cierre de la solución (`cerrar_solucion`): los viajes que quitó el ajuste y los pendientes van a la
      post-reparación, y los depósitos de retorno se reparten con `assign_return_depots`.

    `rng` es un `numpy.random.Generator` creado con la semilla de la corrida (decisión K); `time_limit_s` es el límite de
    cada modelo de un bus (sin límite si es `None`) y `post_repair_time_limit_s`, el de la post-reparación. Con
    `use_preconnection=False` no se forman pares: cada sub-cadena es un viaje suelto (decisión C). Necesita Gurobi."""
    from ebus.heuristics.charger_power import enforce_charging_resources
    from ebus.heuristics.modelo_un_bus import ModeloUnBus, enlaces_de_la_red, opciones_un_bus, red_de_un_bus
    from ebus.heuristics.post_repair import cerrar_solucion

    pares = build_trip_pairs(inst.trips, T_tc) if use_preconnection else []
    pool: dict[str, Trip] = {t.id: t for t in sorted(inst.trips, key=lambda t: (t.start_time, t.id))}
    cadenas: list[TripChain] = []
    pendientes: list[Trip] = []
    libre = CapacidadLibre(inst)
    red = red_de_un_bus(inst, k_sucesores, {(a.id, b.id) for a, b in pares})
    enlaza = enlaces_de_la_red(red, inst)
    modelo = ModeloUnBus(inst, set(), opciones_un_bus(time_limit_s), libre, red)
    inicio = time.perf_counter()
    modelo.build()
    registro.info("Modelo de un bus armado en %.1f s (%d viajes)", time.perf_counter() - inicio, len(inst.trips))
    try:
        while pool:
            bloques = sample_sub_chain(pares, set(pool), N_p, inst, rng, enlaza)
            cadena = None
            while cadena is None:
                modelo.fijar_obligatorios({v.id for bloque in bloques for v in bloque})
                modelo.aplicar_capacidad_libre()
                cadena = modelo.resolver()
                if cadena is not None:
                    break
                if len(bloques) > 1:
                    bloques = bloques[:-1]
                elif len(bloques[0]) > 1:
                    bloques = [bloques[0][:1]]
                else:
                    break
            if cadena is None:
                viaje = bloques[0][0]
                pendientes.append(viaje)
                modelo.quitar_viajes({viaje.id})
                del pool[viaje.id]
                continue
            cadena.bus_id = f"B{len(cadenas) + 1:03d}"
            libre.ocupar(cadena)
            cadenas.append(cadena)
            if len(cadenas) % 50 == 0:
                registro.info("%d buses, %d viajes sin asignar, %.0f s", len(cadenas), len(pool) - len(cadena.trips),
                              time.perf_counter() - inicio)
            modelo.quitar_viajes({v.id for v in cadena.trips})
            for viaje in cadena.trips:
                del pool[viaje.id]
    finally:
        modelo.model.dispose()

    ajustada, quitados = enforce_charging_resources(Solution(chains=cadenas), inst, rng)
    return cerrar_solucion(ajustada, pendientes + quitados, inst, post_repair_time_limit_s)
