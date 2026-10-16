"""Estima la matriz de deadheads de una instancia (paraderos de sus viajes + sus depósitos) y la guarda en CSV.

Ejemplo: python scripts/generar_deadheads.py --config config/instances/VN-dia.yaml
Por defecto escribe en `data.deadhead_file` de la config; no sobrescribe un archivo existente sin --sobrescribir."""

import argparse
from pathlib import Path

from ebus.config import load_config
from ebus.data.deadhead import estimar_matriz_deadheads
from ebus.data.subproblema import RAIZ, depositos_de, leer_viajes
from ebus.model.entities import Station


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None, help="por defecto, data.deadhead_file de la config")
    parser.add_argument("--sobrescribir", action="store_true", help="reemplaza el archivo si ya existe")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data, battery, estimacion = cfg["data"], cfg["battery"], cfg["deadhead_estimation"]
    viajes, terminales, _ = leer_viajes(cfg)
    depositos = depositos_de(cfg, terminales)

    matriz = estimar_matriz_deadheads(
        viajes, [Station(d.id, d.lat, d.lon) for d in depositos],
        velocidad_kmh=estimacion["speed_kmh"], factor_desvio=estimacion["detour_factor"],
        kwh_por_km=battery["consumption_kwh_per_km"] * battery["deadhead_consumption_factor"],
        costo_por_km=cfg["costs"]["deadhead_cost_per_km"], regla_km_viaje=estimacion["trip_km_rule"],
    )
    out = Path(args.out) if args.out else RAIZ / data["deadhead_file"]
    if out.exists() and not args.sobrescribir:
        raise SystemExit(f"{out} ya existe; use --sobrescribir o --out")
    out.parent.mkdir(parents=True, exist_ok=True)
    matriz.to_csv(out, index=False)
    nodos = len(set(matriz["from_id"]))
    print(f"{', '.join(terminales)}: {nodos} nodos ({nodos - len(depositos)} paraderos + depósitos "
          f"{', '.join(d.id for d in depositos)}), {len(matriz)} pares -> {out}")


if __name__ == "__main__":
    main()
