"""Indicadores reportados en la Sección 4 (Tablas 7-8 y análisis de la Fig. 7)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

from ebus.model.entities import Instance, Solution, recharge_events


def peak_concurrent_trips(inst: Instance) -> tuple[int, int]:
    """(máximo de viajes en curso a la vez, minuto en que ocurre). Cada viaje en curso necesita su
    propio bus, así que es una cota inferior del número de buses de cualquier solución."""
    eventos: list[tuple[int, int]] = []
    for trip in inst.trips:
        eventos.append((trip.start_time, 1))
        eventos.append((trip.end_time, -1))
    en_curso, mejor, cuando = 0, 0, 0
    for minuto, delta in sorted(eventos, key=lambda e: (e[0], e[1])):
        en_curso += delta
        if en_curso > mejor:
            mejor, cuando = en_curso, minuto
    return mejor, cuando


def min_buses_con_deadheads(inst: Instance) -> int:
    """Cota inferior del número de buses que considera los deadheads e ignora la batería (Paso 3 de
    docs/planes/plan_solucion_inicial.md): el mínimo de cadenas que cubren todos los viajes si un bus puede hacer `b`
    después de `a` cuando la holgura alcanza para ir de uno al otro, directo o pasando por un depósito (la matriz de
    deadheads no siempre cumple la desigualdad triangular, y una cadena real puede pasar por el depósito a cargar).

    Es la cobertura mínima por caminos del grafo de viajes encadenables: viajes − emparejamiento máximo. Como relaja la
    batería, los cargadores y el horizonte, ninguna solución usa menos buses; y nunca es menor que
    `peak_concurrent_trips`, porque viajes simultáneos no se pueden encadenar.

    El emparejamiento se calcula como un flujo máximo, sin listar los pares encadenables (en `VN-dia` son 23 millones).
    Cada estación inicial tiene una línea de tiempo con dos tipos de eventos: la **llegada** de cada viaje `a` (cuando
    `a` puede estar ahí: su fin + el deadhead mínimo) y la **salida** de cada viaje `b` que parte ahí. Una unidad de flujo
    va de `a` a su llegada a una estación, espera en la línea de tiempo y toma la salida de un `b` posterior. Así, `a`
    alcanza a `b` exactamente cuando `b` es encadenable después de `a`, y el flujo máximo es el emparejamiento máximo.
    A igual minuto, la llegada va antes que la salida (basta con llegar justo a tiempo), salvo la de un viaje a su propia
    estación de salida, que va después: un viaje no se encadena consigo mismo."""
    trips = inst.trips
    n = len(trips)
    if n == 0:
        return 0
    finales = sorted({t.end_station for t in trips})
    iniciales = sorted({t.start_station for t in trips})
    minimo = np.array([[inst.deadhead(e, s) for s in iniciales] for e in finales], dtype=np.int64)
    for d in (d.id for d in inst.depots):
        ida = np.array([inst.deadhead(e, d) for e in finales], dtype=np.int64)
        vuelta = np.array([inst.deadhead(d, s) for s in iniciales], dtype=np.int64)
        minimo = np.minimum(minimo, ida[:, None] + vuelta[None, :])
    fila_final = {e: i for i, e in enumerate(finales)}
    col_inicial = {s: j for j, s in enumerate(iniciales)}

    origen, destino, capacidad = [], [], []

    def arco(u: int, v: int, cap: int) -> None:
        origen.append(u)
        destino.append(v)
        capacidad.append(cap)

    for i in range(n):
        arco(0, 2 + i, 1)
        arco(2 + n + i, 1, 1)
    siguiente = 2 + 2 * n
    for s in iniciales:
        j = col_inicial[s]
        eventos = []
        for i, a in enumerate(trips):
            llegada = a.end_time + int(minimo[fila_final[a.end_station], j])
            propia = a.start_station == s and llegada == a.start_time
            eventos.append((llegada, 2 if propia else 0, i, True))
        eventos.extend((b.start_time, 1, i, False) for i, b in enumerate(trips) if b.start_station == s)
        eventos.sort()
        nodos = range(siguiente, siguiente + len(eventos))
        for nodo, (_, _, i, es_llegada) in zip(nodos, eventos):
            if es_llegada:
                arco(2 + i, nodo, 1)
            else:
                arco(nodo, 2 + n + i, 1)
        for u, v in zip(nodos, nodos[1:]):
            arco(u, v, n)
        siguiente += len(eventos)

    grafo = csr_matrix((np.array(capacidad, dtype=np.int32), (origen, destino)), shape=(siguiente, siguiente))
    emparejados = maximum_flow(grafo, 0, 1, method="dinic").flow_value
    return n - int(emparejados)


@dataclass
class CotaCosto:
    """Cota inferior del costo (1) de cualquier solución (estrategia 1 del costo de la partición, sección 4b de
    docs/planes/entrega_2.md)."""

    buses: int
    costo_fijo: float
    energia_kwh: float
    precio_min_kwh: float
    costo_energia: float

    @property
    def total(self) -> float:
        return self.costo_fijo + self.costo_energia


def cota_costo(inst: Instance, buses: int | None = None) -> CotaCosto:
    """Ninguna solución cuesta menos que `buses × costo fijo + energía mínima × precio más barato`.

    - `buses`: la cota de flota con deadheads (`min_buses_con_deadheads`, que se calcula si no se entrega: en las
      instancias grandes tarda minutos u horas).
    - Energía mínima: cada bus parte con `soc_initial` y termina el día recargado hasta `soc_max` (objetivo (1)), así que
      compra toda la energía que gasta más `soc_max − soc_initial` de su batería. Lo que gasta es al menos la energía de
      los viajes, que es la misma en toda solución; el deadhead y su energía se toman como cero.
    - Toda esa energía se compra a lo más barato al precio mínimo de la tarifa, de día o de noche.

    Ignora la batería, los cargadores y el deadhead, así que puede no ser alcanzable ni con un solver exacto."""
    if buses is None:
        buses = min_buses_con_deadheads(inst)
    bus = inst.bus_type
    energia = sum(t.energy_kwh(bus) for t in inst.trips) + buses * (inst.soc_max - inst.soc_initial) * bus.battery_kwh
    precio = min(inst.tariff.overnight_price_per_kwh, float(np.min(inst.tariff.price_per_kwh_by_minute)))
    return CotaCosto(buses, buses * bus.fixed_cost_per_bus, energia, precio, energia * precio)


def depot_occupancy_profile(sol: Solution, inst: Instance, depot_id: str) -> list[int]:
    """Buses dentro del depósito en cada minuto de `[horizon_start, horizon_end]` (índice 0 = `horizon_start`).

    Un bus está dentro desde el inicio del horizonte hasta que sale hacia su primer viaje, mientras
    carga (desde que llega al depósito hasta que termina la carga) y desde que vuelve al final del día.
    Como todos los buses parten y terminan en el depósito, la ocupación en los extremos del horizonte
    es igual a la flota; lo informativo es el perfil entre medio (ver docs/decisiones/modelo.md)."""
    largo = inst.horizon_end - inst.horizon_start + 1
    delta = [0] * (largo + 1)

    def dentro(desde: int, hasta: int) -> None:
        a, b = max(desde, inst.horizon_start), min(hasta, inst.horizon_end + 1)
        if b > a:
            delta[a - inst.horizon_start] += 1
            delta[b - inst.horizon_start] -= 1

    for chain in sol.chains:
        if not chain.trips:
            continue
        primero, ultimo = chain.trips[0], chain.trips[-1]
        if chain.depot_id == depot_id:
            iniciales = recharge_events(chain, 0)
            carga_inicial = iniciales[-1] if iniciales else None
            salida = primero.start_time - inst.deadhead(chain.depot_id, primero.start_station)
            if carga_inicial is not None and carga_inicial.depot_id == depot_id:
                salida = carga_inicial.end_time
            dentro(inst.horizon_start, salida)
        for k in range(1, len(chain.trips)):
            segmentos = recharge_events(chain, k)
            carga = segmentos[-1] if segmentos else None
            if carga is not None and carga.depot_id == depot_id:
                previo = chain.trips[k - 1]
                dentro(previo.end_time + inst.deadhead(previo.end_station, depot_id), carga.end_time)
        finales = recharge_events(chain, len(chain.trips))
        carga_final = finales[-1] if finales else None
        carga_final_aqui = carga_final is not None and carga_final.depot_id == depot_id
        if carga_final_aqui or chain.return_depot_id == depot_id:
            llegada = ultimo.end_time + inst.deadhead(ultimo.end_station, depot_id)
            if carga_final_aqui:
                dentro(llegada, carga_final.end_time)
                llegada = carga_final.end_time
            if chain.return_depot_id == depot_id:
                dentro(llegada, inst.horizon_end + 1)

    perfil, acumulado = [], 0
    for d in delta[:largo]:
        acumulado += d
        perfil.append(acumulado)
    return perfil


def minutes_over_capacity(perfil: list[int], capacidad: int) -> int:
    """Minutos del perfil en que hay más buses que plazas."""
    return sum(1 for n in perfil if n > capacidad)


def solution_summary(sol: Solution, inst: Instance) -> dict:
    """Costo total y desglose, n° de buses, % tiempo en viaje / vacío / carga, cargas por bus,
    duración media de carga, servicios distintos por bus, buses con vacío pre/post operación."""
    raise NotImplementedError


def charging_load_profile(sol: Solution, inst: Instance) -> pd.DataFrame:
    """Potencia y n° de cargadores en uso por depósito y periodo."""
    raise NotImplementedError
