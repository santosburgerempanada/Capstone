"""Sensibilidad del ALNS a `destroy_max` y `accept_probability` (valores de config/experiments.yaml, sección
`sensitivity`), un parámetro a la vez, partiendo del baseline. Es un experimento: escribe
results/<instancia>/<experimento>/sensibilidad.csv (una fila por corrida) y sensibilidad_resumen.csv.

Ejemplo: python scripts/sensibilidad.py --config config/instances/toy_carga.yaml --experimento E1-sensibilidad
         --iteraciones 3000 --semillas 3"""

import argparse
from pathlib import Path

from ebus.baseline.greedy_construction import build_baseline_solution
from ebus.config import load_config
from ebus.data.build_instance import build_instance
from ebus.evaluation.sensibilidad import resumen_sensibilidad, sensibilidad

PARAMETROS = ("destroy_max", "accept_probability")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--experimentos", default="config/experiments.yaml")
    parser.add_argument("--iteraciones", type=int, required=True)
    parser.add_argument("--semillas", type=int, default=None,
                        help="cuántas semillas (1, 2, ...); por defecto, sensitivity.repetitions")
    parser.add_argument("--experimento", required=True, help="identificador del experimento, p. ej. E1-sensibilidad")
    parser.add_argument("--out-dir", default="results", help="carpeta base de resultados")
    args = parser.parse_args()

    cfg = load_config(args.config)
    variaciones_cfg = load_config(args.experimentos)["sensitivity"]
    inst = build_instance(cfg)
    inicial = build_baseline_solution(inst)
    semillas = list(range(1, (args.semillas or variaciones_cfg["repetitions"]) + 1))
    tabla = sensibilidad(inicial, inst, cfg["alns"], {p: variaciones_cfg[p] for p in PARAMETROS},
                         args.iteraciones, semillas)

    nombre = cfg["case"]["name"]
    carpeta = Path(args.out_dir) / nombre / args.experimento
    carpeta.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(carpeta / "sensibilidad.csv", index=False)
    resumen = resumen_sensibilidad(tabla)
    resumen.to_csv(carpeta / "sensibilidad_resumen.csv")
    print(resumen)


if __name__ == "__main__":
    main()
