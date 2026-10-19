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
    from ebus.heuristics.modelo_un_bus import cadena_de_un_bus

    return cadena_de_un_bus(sub_chain, pool, inst, time_limit_s, libre, k_sucesores, red_completa)


def initial_solution(inst: Instance, T_tc: int, N_p: int, rng: np.random.Generator, time_limit_s: float | None,
                     post_repair_time_limit_s: float, use_preconnection: bool = True,
                     k_sucesores: int | None = None) -> Solution:
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
