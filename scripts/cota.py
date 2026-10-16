"""Calcula la cota inferior de flota y de costo de una instancia y la guarda en results/<instancia>/cota.json
(estrategia 1 del costo de la partición, sección 4b de docs/planes/entrega_2.md).

Ejemplo: python scripts/cota.py --config config/instances/VN-dia.yaml
La cota de flota con deadheads tarda segundos en VN-10, minutos en un terminal y más en las instancias de varios
terminales (38 min en VN-EC-dia)."""

import argparse
import json
import time
from datetime import datetime

from ebus.config import load_config
from ebus.data.build_instance import build_instance
from ebus.evaluation.metrics import cota_costo, min_buses_con_deadheads, peak_concurrent_trips
from ebus.evaluation.resultados import _version_del_codigo, carpeta_de_corrida


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default=None, help="carpeta base de resultados (por defecto, results/)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    inst = build_instance(cfg)
    instancia = cfg["case"]["name"]

    t0 = time.perf_counter()
    buses = min_buses_con_deadheads(inst)
    tiempo_s = time.perf_counter() - t0
    cota = cota_costo(inst, buses=buses)
    pico, minuto = peak_concurrent_trips(inst)

    resumen = {
        "instancia": instancia,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "config": args.config.replace("\\", "/"),
        "codigo": _version_del_codigo(),
        "moneda": cfg.get("costs", {}).get("currency"),
        "viajes": len(inst.trips),
        "depositos": [d.id for d in inst.depots],
        "viajes_simultaneos": {"maximo": pico, "hora": f"{minuto // 60 % 24:02d}:{minuto % 60:02d}"},
        "buses": cota.buses,
        "costo": {"total": round(cota.total, 2), "costo_fijo_buses": round(cota.costo_fijo, 2),
                  "energia": round(cota.costo_energia, 2)},
        "energia_kwh": round(cota.energia_kwh, 2),
        "precio_min_kwh": cota.precio_min_kwh,
        "tiempo_cota_flota_s": round(tiempo_s, 3),
    }
    carpeta = carpeta_de_corrida(instancia, out_dir=args.out_dir)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / "cota.json"
    ruta.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{instancia}: {len(inst.trips)} viajes; cota de flota {cota.buses} buses (viajes simultáneos: {pico}); "
          f"cota de costo {cota.total:,.1f} = {cota.costo_fijo:,.1f} de buses + {cota.costo_energia:,.1f} de energía "
          f"({cota.energia_kwh:,.0f} kWh × {cota.precio_min_kwh}); flota en {tiempo_s:.0f} s -> {ruta}")


if __name__ == "__main__":
    main()
