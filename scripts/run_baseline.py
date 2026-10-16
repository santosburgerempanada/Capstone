"""Corre el caso base (heurística greedy, sin garantía de optimalidad) sobre una instancia y lo guarda en
results/<instancia>/baseline.* (es determinista: uno por instancia)."""

import argparse
import time

from ebus.baseline.greedy_construction import build_baseline_solution
from ebus.config import load_config
from ebus.data.build_instance import build_instance
from ebus.evaluation.feasibility import check_solution
from ebus.evaluation.resultados import guardar_corrida
from ebus.evaluation.metrics import depot_occupancy_profile, minutes_over_capacity, peak_concurrent_trips
from ebus.model.cost import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default=None, help="carpeta base de resultados (por defecto, results/)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    inst = build_instance(cfg)

    t0 = time.perf_counter()
    sol = build_baseline_solution(inst)
    tiempo_s = time.perf_counter() - t0
    breakdown = evaluate(sol, inst)
    violations = check_solution(sol, inst)

    print(f"Costo total: {breakdown.total:.1f} ({cfg.get('costs', {}).get('currency', '?')}, costo del bloque)")
    print(f"  costo fijo buses: {breakdown.bus_fixed_cost:.1f}  deadhead: {breakdown.deadhead:.1f}  "
          f"carga diurna: {breakdown.daytime_charging:.1f}  carga nocturna: {breakdown.overnight_charging:.1f}")
    print(f"Buses: {sol.n_buses}")
    pico, minuto = peak_concurrent_trips(inst)
    print(f"  cota inferior: {pico} viajes simultáneos a las {minuto // 60:02d}:{minuto % 60:02d} "
          f"(ninguna solución puede usar menos buses)")
    for depot in inst.depots:
        perfil = depot_occupancy_profile(sol, inst, depot.id)
        capacidad = f"{depot.capacity}" if depot.capacity is not None else "sin límite"
        print(f"  depósito {depot.id}: capacidad de estacionamiento {capacidad}; buses dentro: "
              f"{perfil[0]} al inicio del horizonte, mínimo {min(perfil)}")
        if depot.capacity is not None:
            exceso = minutes_over_capacity(perfil, depot.capacity)
            print(f"    {exceso} de {len(perfil)} minutos con más buses que la capacidad (el validador la impone)")
    print(f"Violaciones de factibilidad: {len(violations)}")
    for v in violations:
        print(f"  - {v}")

    ruta_json, ruta_csv = guardar_corrida(
        "baseline", cfg["case"]["name"], sol, inst, tiempo_s, out_dir=args.out_dir, cfg=cfg, config_path=args.config
    )
    print(f"Resultados guardados en {ruta_json} y {ruta_csv}")


if __name__ == "__main__":
    main()
