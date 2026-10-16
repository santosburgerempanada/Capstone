"""ALNS sobre una instancia, partiendo del baseline (solución inicial provisoria; ver docs/decisiones/alns.md).

Cada corrida pertenece a un experimento, que se registra en results/<instancia>/README.md (ver
docs/decisiones/resultados.md). Se guarda en results/<instancia>/<experimento>/<variante>_s<semilla>.*: la variante
sale del nombre de la config (VN-dia-d20.yaml -> d20; la config base de la instancia -> base).

Ejemplo: python scripts/run_alns.py --config config/instances/VN-dia-d20.yaml --experimento E2-dmax --semilla 1
Corridas de filtro (pocas iteraciones, no se versionan): --filtro las guarda en <experimento>/filtro/."""

import argparse

from ebus.baseline.greedy_construction import build_baseline_solution
from ebus.config import load_config
from ebus.data.build_instance import build_instance
from ebus.evaluation.feasibility import check_solution
from ebus.evaluation.resultados import guardar_corrida, guardar_historia, variante_por_defecto
from ebus.heuristics.alns import run_alns
from ebus.model.cost import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--iteraciones", type=int, default=None, help="reemplaza alns.max_iterations de la config")
    parser.add_argument("--semilla", type=int, default=None, help="reemplaza alns.seed de la config")
    parser.add_argument("--experimento", required=True,
                        help="identificador del experimento, p. ej. E2-dmax (ver results/<instancia>/README.md)")
    parser.add_argument("--variante", default=None,
                        help="nombre de la variante (por defecto, el de la config sin el de la instancia)")
    parser.add_argument("--filtro", action="store_true",
                        help="corrida de filtro: se guarda en <experimento>/filtro/, que no se versiona")
    parser.add_argument("--out-dir", default=None, help="carpeta base de resultados (por defecto, results/)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    inst = build_instance(cfg)
    inicial = build_baseline_solution(inst)
    resultado = run_alns(inicial, inst, cfg["alns"], args.iteraciones, args.semilla)
    mejor = resultado.mejor
    costo = evaluate(mejor, inst)
    moneda = cfg.get("costs", {}).get("currency", "?")

    print(f"Iteraciones: {len(resultado.historia)} en {resultado.tiempo_s:.1f} s")
    print(f"Costo: {resultado.costo_inicial:.1f} (baseline) -> {costo.total:.1f} {moneda} "
          f"({100 * (costo.total / resultado.costo_inicial - 1):+.2f} %)")
    print(f"  costo fijo buses: {costo.bus_fixed_cost:.1f}  deadhead: {costo.deadhead:.1f}  "
          f"carga diurna: {costo.daytime_charging:.1f}  carga nocturna: {costo.overnight_charging:.1f}")
    print(f"Buses: {inicial.n_buses} (baseline) -> {mejor.n_buses}")
    print(f"Violaciones de factibilidad: {len(check_solution(mejor, inst))}")

    instancia = cfg["case"]["name"]
    semilla = args.semilla if args.semilla is not None else cfg["alns"]["seed"]
    experimento = f"{args.experimento}/filtro" if args.filtro else args.experimento
    nombre = f"{args.variante or variante_por_defecto(args.config, instancia)}_s{semilla}"
    corrida = {
        "iteraciones": len(resultado.historia),
        "semilla": semilla,
        "solucion_inicial": "baseline",
        "costo_inicial": round(resultado.costo_inicial, 2),
        "destroy_max": cfg["alns"]["destroy_max"],
        "accept_probability": cfg["alns"]["accept_probability"],
    }
    ruta_json, ruta_csv = guardar_corrida("alns", instancia, mejor, inst, resultado.tiempo_s, out_dir=args.out_dir,
                                          cfg=cfg, config_path=args.config, corrida=corrida, experimento=experimento,
                                          nombre=nombre)
    ruta_historia = guardar_historia(nombre, instancia, resultado.historia, args.out_dir, experimento)
    print(f"Resultados guardados en {ruta_json}, {ruta_csv} y {ruta_historia}")


if __name__ == "__main__":
    main()
