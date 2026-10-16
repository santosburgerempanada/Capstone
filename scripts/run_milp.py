"""Resuelve el MILP (1)-(16) con Gurobi. Solo para instancias pequeñas (en el paper, <= 105 viajes)."""

import argparse

from ebus.config import load_config
from ebus.data.build_instance import build_instance
from ebus.evaluation.feasibility import check_solution
from ebus.model.cost import evaluate
from ebus.model.milp import EvspMilp, MilpOptions
from ebus.model.network import build_network


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    inst = build_instance(cfg)
    net = build_network(inst)
    milp = EvspMilp(inst, net, MilpOptions(time_limit_s=cfg["solver"]["time_limit_s"], mip_gap=cfg["solver"]["mip_gap"]))
    milp.build()
    sol = milp.solve()
    costs = evaluate(sol, inst)
    print(f"Costo: {sol.cost:.8f}, buses: {sol.n_buses}")
    print(f"Estado Gurobi: {milp.model.Status}, gap: {milp.model.MIPGap:.6g}")
    print(f"Objetivo Gurobi: {milp.model.ObjVal:.8f}; evaluate: {costs.total:.8f}")
    print(f"Desglose: {costs}")
    print(f"Violaciones: {check_solution(sol, inst)}")
    milp.model.dispose()


if __name__ == "__main__":
    main()
