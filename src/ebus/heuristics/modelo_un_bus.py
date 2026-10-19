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
    return MilpOptions(time_limit_s=GRB.INFINITY if time_limit_s is None else time_limit_s, mip_gap=0.0,
                       enforce_charger_limit=False, enforce_power_limit=False, enforce_depot_capacity=False,
                       max_buses=1, output_flag=output_flag)


class ModeloUnBus(EvspMilp):

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
        ocupados = set(viajes) & self.asignados
        if ocupados:
            raise ValueError(f"Viajes obligatorios ya asignados a otro bus: {sorted(ocupados)}")
        self.obligatorios = set(viajes)
        self._actualizar_cobertura()

    def quitar_viajes(self, viajes: set[str]) -> None:
        self.asignados |= set(viajes)
        self.obligatorios -= set(viajes)
        self._actualizar_cobertura()

    def _add_charger_limit(self) -> None:
        pass

    def aplicar_capacidad_libre(self) -> None:
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
