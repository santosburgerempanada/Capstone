"""Event-based time-space network without a vehicle index.

A depot lane belongs to a *commercial arrival*, not to a bus. Only the selected
flow determines which bus uses it. Lanes must not merge: a scalar SOC cannot
represent two buses at the same physical place and time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

from ebus.model.entities import Instance


class Node(NamedTuple):
    location: str
    time: int
    event: str = ""


class ArcType(Enum):
    TRIP = "trip"
    CONNECTION = "conn"
    WAIT = "wait"
    CHARGE = "charge"


@dataclass(frozen=True)
class Arc:
    type: ArcType
    origin: Node
    dest: Node
    trip_id: str | None = None
    deadhead_min: int = 0
    deadhead_distance_km: float = 0.0

    @property
    def duration(self) -> int:
        return self.dest.time - self.origin.time


@dataclass
class TimeSpaceNetwork:
    nodes: set[Node] = field(default_factory=set)
    arcs: list[Arc] = field(default_factory=list)
    source_nodes: set[Node] = field(default_factory=set)
    sink_nodes: set[Node] = field(default_factory=set)

    def arcs_in(self, node: Node) -> list[Arc]:
        return [a for a in self.arcs if a.dest == node]

    def arcs_out(self, node: Node) -> list[Arc]:
        return [a for a in self.arcs if a.origin == node]

    def depot_arcs_covering(self, depot_id: str, t: int) -> list[Arc]:
        return [a for a in self.arcs if a.type in (ArcType.WAIT, ArcType.CHARGE)
                and a.origin.location == depot_id and a.origin.time <= t < a.dest.time]

    def add(self, arc: Arc) -> None:
        self.nodes.update((arc.origin, arc.dest))
        self.arcs.append(arc)


def add_trip_arcs(net: TimeSpaceNetwork, inst: Instance) -> None:
    if len({t.id for t in inst.trips}) != len(inst.trips):
        raise ValueError("Trip IDs must be unique")
    for t in inst.trips:
        if t.end_time <= t.start_time:
            raise ValueError("Commercial trips must have positive duration")
        net.add(Arc(ArcType.TRIP, Node(t.start_station, t.start_time, f"start:{t.id}"),
                    Node(t.end_station, t.end_time, f"end:{t.id}"), t.id))


def _connection(net: TimeSpaceNetwork, inst: Instance, origin: Node, dest: Node) -> None:
    minutes = inst.deadhead(origin.location, dest.location)
    if origin.time + minutes <= dest.time:
        net.add(Arc(ArcType.CONNECTION, origin, dest, deadhead_min=minutes,
                    deadhead_distance_km=inst.deadhead_distance_km(origin.location, dest.location)))


def add_connection_arcs(net: TimeSpaceNetwork, inst: Instance) -> None:
    trips = [a for a in net.arcs if a.type == ArcType.TRIP]
    for a in trips:
        for b in trips:
            if a != b and a.dest.time <= b.origin.time:
                _connection(net, inst, a.dest, b.origin)
        for d in inst.depots:
            departure = a.origin.time - inst.deadhead(d.id, a.origin.location)
            if departure >= inst.horizon_start:
                event = f"source:{d.id}:{a.trip_id}"
                source = Node(d.id, inst.horizon_start, event)
                ready = Node(d.id, departure, event + ":ready")
                net.source_nodes.add(source)
                net.add(Arc(ArcType.WAIT, source, ready))
                _connection(net, inst, ready, a.origin)
            arrival = a.dest.time + inst.deadhead(a.dest.location, d.id)
            if arrival <= inst.horizon_end:
                ready = Node(d.id, arrival, f"arrival:{d.id}:{a.trip_id}")
                _connection(net, inst, a.dest, ready)


def add_waiting_arcs(net: TimeSpaceNetwork, inst: Instance) -> None:
    if inst.time_step <= 0:
        raise ValueError("time_step must be positive")
    trips = [a for a in net.arcs if a.type == ArcType.TRIP]
    arrivals = sorted(n for n in net.nodes if n.event.startswith("arrival:"))
    tariff_changes = {t for t in range(inst.horizon_start + 1, inst.horizon_end)
                      if inst.tariff.price_at(t) != inst.tariff.price_at(t - 1)}
    for arrival in arrivals:
        departures = {a.origin.time - inst.deadhead(arrival.location, a.origin.location): a
                      for a in trips}
        times = {arrival.time, inst.horizon_end}
        times.update(range(arrival.time, inst.horizon_end, inst.time_step))
        times.update(t for t in tariff_changes if arrival.time < t < inst.horizon_end)
        times.update(t for t in departures if arrival.time <= t <= inst.horizon_end)
        nodes = [Node(arrival.location, t, arrival.event) for t in sorted(times)]
        net.sink_nodes.add(nodes[-1])
        for left, right in zip(nodes, nodes[1:]):
            net.add(Arc(ArcType.WAIT, left, right))
        by_time = {n.time: n for n in nodes}
        for a in trips:
            departure = a.origin.time - inst.deadhead(arrival.location, a.origin.location)
            if departure in by_time:
                _connection(net, inst, by_time[departure], a.origin)


def add_charging_arcs(net: TimeSpaceNetwork) -> None:
    for a in list(net.arcs):
        if a.type == ArcType.WAIT and a.origin not in net.source_nodes:
            net.add(Arc(ArcType.CHARGE, a.origin, a.dest))


def merge_nodes(net: TimeSpaceNetwork) -> None:
    """Canonicalize identical logical events, never different arrival lanes.

    Physical-coordinate merging would permit exchanging SOC between buses.
    Trip endpoints already have one node shared by all admissible connections.
    """
    net.arcs = list(dict.fromkeys(net.arcs))
    net.nodes = {n for a in net.arcs for n in (a.origin, a.dest)}
    net.nodes.update(net.source_nodes | net.sink_nodes)


def build_network(inst: Instance) -> TimeSpaceNetwork:
    net = TimeSpaceNetwork()
    add_trip_arcs(net, inst)
    add_connection_arcs(net, inst)
    add_waiting_arcs(net, inst)
    add_charging_arcs(net)
    merge_nodes(net)
    return net
