from __future__ import annotations

from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB

from ebus.model.entities import Instance, Solution, TripChain, RechargeEvent
from ebus.model.network import ArcType, TimeSpaceNetwork


@dataclass
class MilpOptions:
    time_limit_s: float = 14400
    mip_gap: float = 0.0
    enforce_charger_limit: bool = True
    enforce_power_limit: bool = True
    enforce_depot_capacity: bool = True
    big_m: float = 1e4
    max_buses: int | None = None
    output_flag: bool = True


class EvspMilp:
    def __init__(self, inst: Instance, net: TimeSpaceNetwork, options: MilpOptions | None = None):
        self.inst, self.net = inst, net
        self.options = options or MilpOptions()
        self.model = None
        self.incoming = {n: [] for n in net.nodes}
        self.outgoing = {n: [] for n in net.nodes}
        for a in net.arcs:
            self.incoming[a.dest].append(a)
            self.outgoing[a.origin].append(a)

    def build(self) -> None:
        if self.model is not None:
            raise RuntimeError("Model already built")
        self._add_variables()
        self._set_objective()
        self._add_coverage()
        self._add_flow_conservation()
        self._add_single_inflow()
        self._add_wait_charge_exclusion()
        self._add_depot_balance()
        self._add_depot_capacity()
        self._add_charger_limit()
        self._add_power_allocation()
        self._add_soc_bounds()
        self._add_soc_dynamics()
        self._add_initial_conditions()
        self.model.update()

    def _add_variables(self) -> None:
        self.model = gp.Model("ebus")
        self.model.Params.OutputFlag = int(self.options.output_flag)
        self.model.Params.TimeLimit = self.options.time_limit_s
        self.model.Params.MIPGap = self.options.mip_gap
        self.model.Params.FeasibilityTol = 1e-8
        self.model.Params.IntFeasTol = 1e-8
        self.y = {a: self.model.addVar(vtype=GRB.BINARY, name=f"{a.type.value}[{i}]")
                  for i, a in enumerate(self.net.arcs)}
        self.X, self.B, self.W, self.C = ({a: self.y[a] for a in self.net.arcs if a.type == kind}
                                         for kind in ArcType)
        self.s = {n: self.model.addVar(ub=self.inst.soc_max, name=f"s[{i}]")
                  for i, n in enumerate(sorted(self.net.nodes))}
        self.v = {n: gp.quicksum(self.y[a] for a in
                               (self.outgoing[n] if n in self.net.source_nodes else self.incoming[n]))
                  for n in self.net.nodes}
        self.p = {a: self.model.addVar(ub=self.inst.bus_type.max_charge_kw, name=f"p[{i}]")
                  for i, a in enumerate(self.C)}
        self.q = {a: self.model.addVar(name=f"q[{i}]") for i, a in enumerate(self.C)}
        self.n_buses = gp.quicksum(self.v[n] for n in self.net.source_nodes)

    def _set_objective(self) -> None:
        inst, bus = self.inst, self.inst.bus_type
        self.objective_components = {
            "bus_fixed_cost": bus.fixed_cost_per_bus * self.n_buses,
            "deadhead": inst.deadhead_cost_per_km * gp.quicksum(a.deadhead_distance_km * b for a, b in self.B.items()),
            "daytime_charging": gp.quicksum(inst.tariff.price_at(a.origin.time) * q for a, q in self.q.items()),
            "overnight_charging": bus.battery_kwh * inst.tariff.overnight_price_per_kwh * gp.quicksum(
                inst.soc_max * self.v[n] - self.s[n] for n in self.net.sink_nodes),
        }
        self.model.setObjective(gp.quicksum(self.objective_components.values()), GRB.MINIMIZE)

    def _add_coverage(self) -> None:
        for trip in self.inst.trips:
            self.model.addConstr(gp.quicksum(x for a, x in self.X.items() if a.trip_id == trip.id) == 1,
                                 name=f"coverage[{trip.id}]")

    def _add_flow_conservation(self) -> None:
        for n in sorted(self.net.nodes - self.net.source_nodes - self.net.sink_nodes):
            self.model.addConstr(gp.quicksum(self.y[a] for a in self.incoming[n]) ==
                                 gp.quicksum(self.y[a] for a in self.outgoing[n]), name="flow")

    def _add_single_inflow(self) -> None:
        for n in sorted(self.net.nodes):
            arcs = self.outgoing[n] if n in self.net.source_nodes else self.incoming[n]
            paired = (len(arcs) == 2 and {a.type for a in arcs} == {ArcType.WAIT, ArcType.CHARGE}
                      and arcs[0].origin == arcs[1].origin and arcs[0].dest == arcs[1].dest)
            if len(arcs) > 1 and not paired:
                self.model.addConstr(self.v[n] <= 1, name="single_flow")

    def _add_wait_charge_exclusion(self) -> None:
        wait = {(a.origin, a.dest): w for a, w in self.W.items()}
        for a, c in self.C.items():
            self.model.addConstr(wait[a.origin, a.dest] + c <= 1, name="wait_or_charge")

    def _add_depot_balance(self) -> None:
        for d in self.inst.depots:
            self.model.addConstr(gp.quicksum(self.v[n] for n in self.net.source_nodes if n.location == d.id) ==
                                 gp.quicksum(self.v[n] for n in self.net.sink_nodes if n.location == d.id),
                                 name=f"balance[{d.id}]")
        if self.options.max_buses is not None:
            if self.options.max_buses < 0:
                raise ValueError("max_buses must be nonnegative")
            self.model.addConstr(self.n_buses <= self.options.max_buses, name="fleet_limit")

    def _add_depot_capacity(self) -> None:
        if not self.options.enforce_depot_capacity:
            return
        for d in self.inst.depots:
            if d.capacity is None:
                continue
            arcos = [(a, y) for a, y in self.y.items()
                     if a.type in (ArcType.WAIT, ArcType.CHARGE) and a.origin.location == d.id]
            for t in sorted({a.origin.time for a, _ in arcos}):
                dentro = [y for a, y in arcos if a.origin.time <= t < a.dest.time]
                if len(dentro) > d.capacity:
                    self.model.addConstr(gp.quicksum(dentro) <= d.capacity, name=f"capacidad[{d.id},{t}]")

    def _resource_intervals(self):
        for d in self.inst.depots:
            arcs = [a for a in self.C if a.origin.location == d.id]
            points = sorted({t for a in arcs for t in (a.origin.time, a.dest.time)})
            for t in points[:-1]:
                active = [a for a in arcs if a.origin.time <= t < a.dest.time]
                if active:
                    yield d, t, active

    def _add_charger_limit(self) -> None:
        if self.options.enforce_charger_limit:
            for d, t, active in self._resource_intervals():
                self.model.addConstr(gp.quicksum(self.C[a] for a in active) <= d.n_chargers,
                                     name=f"chargers[{d.id},{t}]")

    def _add_power_allocation(self) -> None:
        for a, c in self.C.items():
            self.model.addConstr(self.p[a] <= self.inst.bus_type.max_charge_kw * c, name="individual_power")
            self.model.addConstr(self.q[a] == self.p[a] * a.duration / 60, name="energy_power_link")
        if self.options.enforce_power_limit:
            for d, t, active in self._resource_intervals():
                self.model.addConstr(gp.quicksum(self.p[a] for a in active) <= d.power_kw,
                                     name=f"power[{d.id},{t}]")

    def _add_soc_bounds(self) -> None:
        for n in sorted(self.net.nodes):
            self.model.addConstr(self.s[n] >= self.inst.soc_min * self.v[n], name="soc_min")
            self.model.addConstr(self.s[n] <= self.inst.soc_max * self.v[n], name="soc_max")

    def _add_soc_dynamics(self) -> None:
        trips = {t.id: t for t in self.inst.trips}
        bus = self.inst.bus_type
        charges = {(a.origin, a.dest): a for a in self.C}
        for a in self.net.arcs:
            if a.type == ArcType.CHARGE:
                continue
            energy = 0.0
            selected = self.y[a]
            if a.type == ArcType.TRIP:
                energy = trips[a.trip_id].distance_km * bus.trip_kwh_per_km
            elif a.type == ArcType.CONNECTION:
                energy = a.deadhead_distance_km * bus.deadhead_kwh_per_km
            change = -energy / bus.battery_kwh
            if a.type == ArcType.WAIT and (a.origin, a.dest) in charges:
                charge = charges[a.origin, a.dest]
                selected = selected + self.C[charge]
                change = self.q[charge] / bus.battery_kwh
            big_m = self.inst.soc_max + energy / bus.battery_kwh
            if big_m > self.options.big_m:
                raise ValueError("MilpOptions.big_m is insufficient for an arc")
            residual = self.s[a.dest] - self.s[a.origin] - change
            self.model.addConstr(residual <= big_m * (1 - selected), name="soc_dynamics_upper")
            self.model.addConstr(residual >= -big_m * (1 - selected), name="soc_dynamics_lower")

    def _add_initial_conditions(self) -> None:
        for n in sorted(self.net.source_nodes):
            self.model.addConstr(self.s[n] == self.inst.soc_initial * self.v[n], name="initial_soc")

    def solve(self) -> Solution:
        if self.model is None:
            self.build()
        self.model.optimize()
        if not self.model.SolCount:
            raise RuntimeError(f"Gurobi returned no incumbent (status={self.model.Status})")
        return self.extract_solution()

    def extract_solution(self) -> Solution:
        from ebus.model.cost import evaluate
        from ebus.evaluation.feasibility import check_solution

        solution = Solution(chains=self.extraer_cadenas())
        violations = check_solution(solution, self.inst)
        if violations:
            raise RuntimeError("Extracted solution is infeasible: " + "; ".join(violations))
        solution.cost = evaluate(solution, self.inst).total
        tolerance = 1e-5 + 1e-7 * abs(solution.cost)
        if abs(solution.cost - self.model.ObjVal) > tolerance:
            raise RuntimeError(f"Objective mismatch: Gurobi={self.model.ObjVal}, evaluate={solution.cost}")
        return solution

    def extraer_cadenas(self) -> list[TripChain]:
        if self.model is None or not self.model.SolCount:
            raise RuntimeError("Cannot extract without an incumbent")
        trips = {t.id: t for t in self.inst.trips}
        active = {a for a, y in self.y.items() if y.X > .5}
        visited, chains = set(), []
        for source in sorted(self.net.source_nodes):
            if self.v[source].getValue() < .5:
                continue
            chain = TripChain(depot_id=source.location, bus_id=f"B{len(chains)+1:03d}")
            node = source
            while node not in self.net.sink_nodes:
                choices = [a for a in self.outgoing[node] if a in active]
                if len(choices) != 1 or choices[0] in visited:
                    raise RuntimeError("Selected flow is not a collection of unit paths")
                a = choices[0]
                visited.add(a)
                if a.type == ArcType.TRIP:
                    chain.trips.append(trips[a.trip_id])
                    chain.soc_before.append(self.s[a.origin].X)
                    chain.soc_after.append(self.s[a.dest].X)
                elif a.type == ArcType.CONNECTION and a.dest.event.startswith("arrival:"):
                    chain.recharges.setdefault(len(chain.trips), []).append(
                        RechargeEvent(a.dest.location, a.dest.time, a.dest.time, 0.0))
                elif a.type == ArcType.CHARGE:
                    events = chain.recharges.setdefault(len(chain.trips), [])
                    event = RechargeEvent(a.origin.location, a.origin.time, a.dest.time, max(0.0, self.p[a].X))
                    if events and events[-1].duration == 0 and events[-1].power_kw == 0:
                        events.pop()
                    events.append(event)
                node = a.dest
            chain.return_depot_id = node.location
            chains.append(chain)
        if visited != active:
            raise RuntimeError("Selected arcs remain outside extracted paths")
        return chains
