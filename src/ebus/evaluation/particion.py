from __future__ import annotations

import time

from ebus.model.entities import Instance, Solution

_ESTADOS = {2: "OPTIMAL", 3: "INFEASIBLE", 9: "TIME_LIMIT", 11: "INTERRUPTED"}
_OPTIMO = 2


def resolver_exacto(inst: Instance, limite_s: float, mip_gap: float, output_flag: bool = True
                    ) -> tuple[Solution | None, dict]:
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
