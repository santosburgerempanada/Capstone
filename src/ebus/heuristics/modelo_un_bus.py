"""Modelo (17)-(19) de la solución inicial (Sección 3.1): los viajes que hace **un** bus, sobre la red espacio-tiempo y el
MILP de `model/milp.py` (decisión A de docs/decisiones/alns.md). Va en un módulo aparte porque necesita Gurobi; la solución
inicial lo importa solo cuando lo usa.

Diferencias con el MILP completo:
- **Objetivo (17) con desempate (D):** maximizar los viajes del bus y, a igual número, el menor costo de deadhead (km).
  Se implementa con un peso por viaje mayor que cualquier costo de deadhead posible de un bus (lexicográfico).
- **Cobertura (18):** los viajes obligatorios (la sub-cadena) se hacen exactamente una vez; los demás, a lo más una vez;
  los ya asignados a otros buses, ninguna.
- **Un bus (19):** `MilpOptions.max_buses = 1`. Con un bus, la restricción (6) lo obliga a volver a su depósito de salida
  (decisión I).
- **Cargadores y potencia del depósito (7)-(8) sobre la capacidad que dejaron libre los buses anteriores** (`libre`,
  decisión E cambiada el 2026-10-11): un arco de carga se prohíbe si en algún minuto de su intervalo no queda un cargador
  libre (o no queda potencia), y su potencia se limita a la potencia libre mínima del intervalo. Con un solo bus es
  exactamente (7)-(8) sobre la capacidad residual. Sin `libre` no se revisan (como el paper).
La carga es la de la red (decisión H): arcos de carga en el depósito, potencia variable y carga parcial.

**Red de un bus (decisión G):** por defecto el modelo usa `red_de_un_bus`, más chica que la del MILP completo: una sola
pista por depósito (en vez de una por cada llegada) y, si se da `k_sucesores`, conexiones directas viaje → viaje solo hacia
los K primeros sucesores alcanzables de cada viaje. Con `red_completa=True` usa la red del MILP (para comparar).

**Armar una vez, resolver muchas (decisión G, 2026-10-12):** `ModeloUnBus` se construye con todos los viajes y, para cada
bus, se cambian solo la cobertura (`fijar_obligatorios`, `quitar_viajes`) y la capacidad libre (`aplicar_capacidad_libre`)
antes de `resolver`. Construir el modelo en Python es lo que más tarda."""

from __future__ import annotations

import bisect
from dataclasses import replace

import gurobipy as gp
from gurobipy import GRB

from ebus.evaluation.feasibility import check_solution
from ebus.heuristics.capacidad import CapacidadLibre
from ebus.model.entities import Instance, Solution, Trip, TripChain
from ebus.model.milp import EvspMilp, MilpOptions
from ebus.model.network import (
    Arc, ArcType, Node, TimeSpaceNetwork, _connection, add_charging_arcs, add_trip_arcs, build_network, merge_nodes,
)

_LIMITES_DEL_DEPOSITO = ("cargas simultáneas", "potencia total")


def red_de_un_bus(inst: Instance, k_sucesores: int | None = None,
                  conexiones_forzadas: set[tuple[str, str]] | None = None) -> TimeSpaceNetwork:
    """Red espacio-tiempo para **un** bus (decisión G). Igual a la del MILP (`model/network.py`) salvo en dos puntos:

    1. **Una sola pista por depósito.** El MILP tiene una pista por cada llegada a un depósito para que el SOC de buses
       distintos no se mezcle; con un bus no hace falta. La pista tiene un nodo en cada llegada y cada salida posibles,
       en los cambios de tarifa, cada `time_step` y al final del horizonte (el sumidero). Entre dos nodos seguidos se puede
       esperar o cargar. Es al menos tan flexible como las pistas del MILP: entre la llegada y la salida del bus, todos los
       tramos de la pista se pueden cargar.
    2. **Conexiones directas viaje → viaje** solo hacia los `k_sucesores` primeros viajes alcanzables de cada viaje
       (ordenados por hora de salida; alcanzable = la holgura alcanza para el deadhead). Sin límite si es `None`. Por el
       depósito, el bus siempre puede tomar cualquier viaje posterior. Las `conexiones_forzadas` (pares de ids de viajes,
       p. ej. los pares pre-conectados de la Sección 3.1) se agregan siempre, estén o no entre los K primeros.

    Los nodos de la pista llevan el evento `arrival:<depósito>:pista`, para que la extracción de cadenas registre las
    visitas al depósito igual que en el MILP. Así, la parte del depósito crece linealmente con los viajes, no con su
    cuadrado."""
    if inst.time_step <= 0:
        raise ValueError("time_step debe ser positivo")
    if k_sucesores is not None and k_sucesores < 0:
        raise ValueError("k_sucesores debe ser no negativo")
    net = TimeSpaceNetwork()
    add_trip_arcs(net, inst)
    viajes = sorted((a for a in net.arcs if a.type == ArcType.TRIP), key=lambda a: (a.origin.time, a.trip_id))
    salidas_viajes = [a.origin.time for a in viajes]

    por_id = {a.trip_id: a for a in viajes}
    for i, j in conexiones_forzadas or ():
        if i in por_id and j in por_id:
            _connection(net, inst, por_id[i].dest, por_id[j].origin)
    for a in viajes:
        agregadas = 0
        for b in viajes[bisect.bisect_left(salidas_viajes, a.dest.time):]:
            if k_sucesores is not None and agregadas >= k_sucesores:
                break
            if b is not a and a.dest.time + inst.deadhead(a.dest.location, b.origin.location) <= b.origin.time:
                _connection(net, inst, a.dest, b.origin)
                agregadas += 1
        for d in inst.depots:
            salida = a.origin.time - inst.deadhead(d.id, a.origin.location)
            if salida >= inst.horizon_start:
                evento = f"source:{d.id}:{a.trip_id}"
                fuente = Node(d.id, inst.horizon_start, evento)
                lista = Node(d.id, salida, evento + ":ready")
                net.source_nodes.add(fuente)
                net.add(Arc(ArcType.WAIT, fuente, lista))
                _connection(net, inst, lista, a.origin)

    cambios_tarifa = {t for t in range(inst.horizon_start + 1, inst.horizon_end)
                      if inst.tariff.price_at(t) != inst.tariff.price_at(t - 1)}
    for d in inst.depots:
        llegadas = [(a, a.dest.time + inst.deadhead(a.dest.location, d.id)) for a in viajes]
        llegadas = [(a, t) for a, t in llegadas if t <= inst.horizon_end]
        if not llegadas:
            continue
        primera = min(t for _, t in llegadas)
        salidas = [(b, b.origin.time - inst.deadhead(d.id, b.origin.location)) for b in viajes]
        salidas = [(b, t) for b, t in salidas if primera <= t <= inst.horizon_end]
        tiempos = {inst.horizon_end} | {t for _, t in llegadas} | {t for _, t in salidas}
        tiempos.update(range(primera, inst.horizon_end, inst.time_step))
        tiempos.update(t for t in cambios_tarifa if primera < t < inst.horizon_end)
        nodos = {t: Node(d.id, t, f"arrival:{d.id}:pista") for t in sorted(tiempos)}
        orden = list(nodos.values())
        net.sink_nodes.add(orden[-1])
        for izquierda, derecha in zip(orden, orden[1:]):
            net.add(Arc(ArcType.WAIT, izquierda, derecha))
        for a, t in llegadas:
            _connection(net, inst, a.dest, nodos[t])
        for b, t in salidas:
            _connection(net, inst, nodos[t], b.origin)
    add_charging_arcs(net)
    merge_nodes(net)
    return net


def enlaces_de_la_red(red: TimeSpaceNetwork, inst: Instance):
    """Función `enlaza(a, b)`: el bus de `red` puede hacer `b` después de `a`, por una conexión directa de la red o
    pasando por un depósito (la pista permite tomar cualquier viaje posterior si la holgura alcanza para ir y volver).
    La usa la sub-cadena para elegir solo pares que caben en la red recortada."""
    directas = {(a.origin.event[len("end:"):], a.dest.event[len("start:"):]) for a in red.arcs
                if a.type == ArcType.CONNECTION and a.origin.event.startswith("end:") and a.dest.event.startswith("start:")}
    depositos = [d.id for d in inst.depots]

    def enlaza(a: Trip, b: Trip) -> bool:
        if (a.id, b.id) in directas:
            return True
        return any(b.start_time - a.end_time >= inst.deadhead(a.end_station, d) + inst.deadhead(d, b.start_station)
                   for d in depositos)

    return enlaza


def opciones_un_bus(time_limit_s: float | None, output_flag: bool = False) -> MilpOptions:
    """Opciones del MILP para un bus: `max_buses = 1` (19), sin los límites de cargadores y potencia del MILP (se aplican
    sobre la capacidad libre) y sin límite de tiempo si `time_limit_s` es `None`. Tampoco impone la capacidad de
    estacionamiento: la solución inicial del paper no se usa en las instancias reales (docs/decisiones/alns.md, 2026-10-12)
    y, con un bus, solo importaría sumada a los buses anteriores."""
    return MilpOptions(time_limit_s=GRB.INFINITY if time_limit_s is None else time_limit_s, mip_gap=0.0,
                       enforce_charger_limit=False, enforce_power_limit=False, enforce_depot_capacity=False,
                       max_buses=1, output_flag=output_flag)


class ModeloUnBus(EvspMilp):
    """`EvspMilp` restringido a un bus: cambia el objetivo, la cobertura y los límites del depósito. Se construye una vez
    y se resuelve para cada bus cambiando la cobertura y la capacidad libre."""

    def __init__(self, inst: Instance, obligatorios: set[str], options: MilpOptions,
                 libre: CapacidadLibre | None = None, red: TimeSpaceNetwork | None = None):
        super().__init__(inst, red if red is not None else red_de_un_bus(inst), options)
        self.obligatorios = set(obligatorios)
        self.asignados: set[str] = set()
        self.libre = libre
        self.cobertura: dict[str, gp.Constr] = {}

    def build(self) -> None:
        super().build()
        self.aplicar_capacidad_libre()
        self.model.update()

    def _set_objective(self) -> None:
        costo = {a: self.inst.deadhead_cost_per_km * a.deadhead_distance_km for a in self.B}
        peso = 1.0 + 2 * (len(self.inst.trips) + 1) * max(costo.values(), default=0.0)
        self.objective_components = {
            "viajes": gp.quicksum(self.X.values()),
            "deadhead": gp.quicksum(costo[a] * b for a, b in self.B.items()),
        }
        self.model.setObjective(peso * self.objective_components["viajes"] - self.objective_components["deadhead"],
                                GRB.MAXIMIZE)

    def _add_coverage(self) -> None:
        por_viaje: dict[str, list] = {}
        for a, x in self.X.items():
            por_viaje.setdefault(a.trip_id, []).append(x)
        for trip in self.inst.trips:
            hecho = gp.quicksum(por_viaje.get(trip.id, []))
            if trip.id in self.obligatorios:
                restriccion = self.model.addConstr(hecho == 1, name=f"cobertura[{trip.id}]")
            else:
                restriccion = self.model.addConstr(hecho <= 1, name=f"cobertura[{trip.id}]")
            self.cobertura[trip.id] = restriccion

    def _actualizar_cobertura(self) -> None:
        for viaje, restriccion in self.cobertura.items():
            if viaje in self.asignados:
                restriccion.Sense, restriccion.RHS = GRB.LESS_EQUAL, 0.0
            elif viaje in self.obligatorios:
                restriccion.Sense, restriccion.RHS = GRB.EQUAL, 1.0
            else:
                restriccion.Sense, restriccion.RHS = GRB.LESS_EQUAL, 1.0

    def fijar_obligatorios(self, viajes: set[str]) -> None:
        """Los viajes de la sub-cadena (18). Deben estar sin asignar."""
        ocupados = set(viajes) & self.asignados
        if ocupados:
            raise ValueError(f"Viajes obligatorios ya asignados a otro bus: {sorted(ocupados)}")
        self.obligatorios = set(viajes)
        self._actualizar_cobertura()

    def quitar_viajes(self, viajes: set[str]) -> None:
        """Viajes asignados a otro bus: este bus no puede hacerlos."""
        self.asignados |= set(viajes)
        self.obligatorios -= set(viajes)
        self._actualizar_cobertura()

    def _add_charger_limit(self) -> None:
        """Los cargadores se aplican sobre la capacidad libre, en `aplicar_capacidad_libre`."""

    def aplicar_capacidad_libre(self) -> None:
        """(7)-(8) sobre la capacidad libre: un bus usa a lo más un cargador a la vez, así que basta con prohibir los arcos
        de carga en cuyo intervalo, en algún minuto, no queda un cargador o potencia libre, y limitar la potencia de cada
        arco a la potencia libre mínima de su intervalo (además del máximo del bus). Sin `libre`, solo el máximo del bus."""
        maximo_bus = self.inst.bus_type.max_charge_kw
        for a, c in self.C.items():
            if self.libre is None:
                c.UB, self.p[a].UB = 1.0, maximo_bus
                continue
            d, desde, hasta = a.origin.location, a.origin.time, a.dest.time
            potencia = self.libre.potencia_libre(d, desde, hasta)
            disponible = self.libre.cargadores_libres(d, desde, hasta) >= 1 and potencia > 0
            c.UB = 1.0 if disponible else 0.0
            self.p[a].UB = min(maximo_bus, max(0.0, potencia))

    def resolver(self) -> TripChain | None:
        """Resuelve el modelo con la cobertura y la capacidad actuales y devuelve la cadena del bus, o `None` si no hay
        solución (los obligatorios no caben en un bus, decisión J, o Gurobi no encontró ninguna en el tiempo dado). La
        cadena pasa el validador sobre sus propios viajes, salvo los límites del depósito (los controla
        `CapacidadLibre.ocupar`); cualquier otra violación es un bug."""
        self.model.optimize()
        if self.model.SolCount == 0:
            return None
        cadenas = self.extraer_cadenas()
        if len(cadenas) != 1:
            raise RuntimeError(f"El modelo de un bus devolvió {len(cadenas)} cadenas")
        cadena = cadenas[0]
        cadena.bus_id = None
        violaciones = [v for v in check_solution(Solution(chains=[cadena]), replace(self.inst, trips=list(cadena.trips)))
                       if not any(limite in v for limite in _LIMITES_DEL_DEPOSITO)]
        if violaciones:
            raise RuntimeError("La cadena del modelo de un bus no es válida: " + "; ".join(violaciones))
        return cadena


def cadena_de_un_bus(obligatorios: list[Trip], pool: list[Trip], inst: Instance, time_limit_s: float | None,
                     libre: CapacidadLibre | None = None, k_sucesores: int | None = None,
                     red_completa: bool = False, output_flag: bool = False) -> TripChain | None:
    """Arma y resuelve una vez el modelo (17)-(19) con los viajes de `pool` (que incluye a los obligatorios). Para muchos
    buses seguidos conviene `ModeloUnBus` armado una vez. Usa `red_de_un_bus` con `k_sucesores`, o la red del MILP con
    `red_completa=True`."""
    ids_pool = {t.id for t in pool}
    faltan = [t.id for t in obligatorios if t.id not in ids_pool]
    if faltan:
        raise ValueError(f"Viajes obligatorios fuera del pool: {faltan}")
    inst_pool = replace(inst, trips=list(pool))
    red = build_network(inst_pool) if red_completa else red_de_un_bus(inst_pool, k_sucesores)
    modelo = ModeloUnBus(inst_pool, {t.id for t in obligatorios}, opciones_un_bus(time_limit_s, output_flag), libre, red)
    modelo.build()
    try:
        return modelo.resolver()
    finally:
        modelo.model.dispose()
