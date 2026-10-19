import argparse
from pathlib import Path

import pandas as pd

from ebus.evaluation.comparison import actualizar_registro, tabla_de_resultados


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instancia", required=True, help="nombre de la instancia (case.name de su config)")
    parser.add_argument("--experimento", default=None, help="solo las corridas de ese experimento (y el baseline)")
    parser.add_argument("--resultados", default="results", help="carpeta base de resultados")
    parser.add_argument("--out", default=None,
                        help="por defecto, <resultados>/<instancia>/comparacion[_<experimento>].csv")
    args = parser.parse_args()

    carpeta = Path(args.resultados) / args.instancia
    tabla = tabla_de_resultados(args.instancia, args.resultados, args.experimento)
    sufijo = f"_{args.experimento}" if args.experimento else ""
    out = Path(args.out or carpeta / f"comparacion{sufijo}.csv")
    tabla.to_csv(out)
    todas = tabla_de_resultados(args.instancia, args.resultados)
    todas = todas[~todas["experimento"].str.contains("filtro")]
    registro = actualizar_registro(carpeta / "README.md", args.instancia, todas)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(tabla)
    print(f"Tabla guardada en {out}; registro actualizado en {registro}")


if __name__ == "__main__":
    main()
