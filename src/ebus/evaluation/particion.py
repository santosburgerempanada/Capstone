"""Costo de la partición en optimalidad, medido de forma exacta en instancias chicas (estrategia 2 de la sección 4b de
docs/planes/entrega_2.md): el MILP resuelve cada terminal por separado y los terminales juntos, y la pérdida es la
diferencia entre la suma de los óptimos por separado y el óptimo del conjunto."""

from __future__ import annotations

import time

from ebus.model.entities import Instance, Solution

_ESTADOS = {2: "OPTIMAL", 3: "INFEASIBLE", 9: "TIME_LIMIT", 11: "INTERRUPTED"}
_OPTIMO = 2


def resolver_exacto(inst: Instance, limite_s: float, mip_gap: float, output_flag: bool = True
                    ) -> tuple[Solution | None, dict]:
    """Resuelve el MILP de `inst` con Gurobi. Devuelve la mejor solución encontrada (None si no hay ninguna) y un resumen:
    estado, si probó el óptimo, objetivo, cota de Gurobi (ningún costo es menor), gap, tamaño del modelo y tiempos. La
    solución sale de `EvspMilp.extract_solution`, que la valida y comprueba que su costo sea el objetivo de Gurobi."""
    from ebus.model.milp import EvspMilp, MilpOptions
    from ebus.model.network import build_network

    t0 = time.perf_counter()
    milp = EvspMilp(inst, build_network(inst), MilpOptions(time_limit_s=limite_s, mip_gap=mip_gap,
                                                            output_flag=output_flag))
    milp.build()
    construccion_s = time.perf_counter() - t0
    modelo = milp.model
    modelo.optimize()

    hay_solucion = modelo.SolCount > 0
    try:
        cota = float(modelo.ObjBound)
    except AttributeError:
        cota = None
    resumen = {
        "estado": _ESTADOS.get(modelo.Status, str(modelo.Status)),
        "optimo": modelo.Status == _OPTIMO,
        "objetivo": float(modelo.ObjVal) if hay_solucion else None,
        "cota_gurobi": cota,
        "gap": float(modelo.MIPGap) if hay_solucion else None,
        "variables": modelo.NumVars,
        "restricciones": modelo.NumConstrs,
        "tiempo_construccion_s": round(construccion_s, 3),
        "tiempo_resolucion_s": round(modelo.Runtime, 3),
        "limite_s": limite_s,
    }
    sol = milp.extract_solution() if hay_solucion else None
    modelo.dispose()
    return sol, resumen


def perdida_particion(separados: list[dict], conjunto: dict) -> dict:
    """Costo de la partición: lo que cuestan los terminales resueltos por separado menos lo que cuestan juntos.

    Cada resultado es el resumen de `resolver_exacto` más `buses` (los de su mejor solución). Si todos llegaron al
    óptimo, la pérdida es exacta; si no, queda en el intervalo [suma de cotas por separado − objetivo del conjunto, suma de
    objetivos por separado − cota del conjunto], porque cada óptimo está entre su cota y su objetivo. Los % son respecto
    del objetivo del conjunto. La diferencia de buses es la de las mejores soluciones (el MILP minimiza el costo, no los
    buses)."""
    todos = [*separados, conjunto]
    if any(r.get("objetivo") is None or r.get("cota_gurobi") is None for r in todos):
        return {"exacta": False, "costo_min": None, "costo_max": None, "pct_min": None, "pct_max": None,
                "buses_separados": None, "buses_conjunto": None, "buses_menos": None}
    costo_min = sum(r["cota_gurobi"] for r in separados) - conjunto["objetivo"]
    costo_max = sum(r["objetivo"] for r in separados) - conjunto["cota_gurobi"]
    exacta = all(r["optimo"] for r in todos)
    if exacta:
        costo_min = costo_max = sum(r["objetivo"] for r in separados) - conjunto["objetivo"]
    buses_separados = sum(r["buses"] for r in separados)
    return {
        "exacta": exacta,
        "costo_min": costo_min, "costo_max": costo_max,
        "pct_min": 100 * costo_min / conjunto["objetivo"], "pct_max": 100 * costo_max / conjunto["objetivo"],
        "buses_separados": buses_separados, "buses_conjunto": conjunto["buses"],
        "buses_menos": buses_separados - conjunto["buses"],
    }
