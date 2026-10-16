"""Costo de la partición medido de forma exacta en instancias chicas (estrategia 2 de la sección 4b de
docs/planes/entrega_2.md).

Resuelve con el MILP cada instancia de `particion_chica.instancias` (una muestra al azar de los viajes de cada terminal,
`data.muestra`) y calcula la pérdida de cada conjunto contra sus partes por separado (`particion_chica.comparaciones`).

Ejemplo: python scripts/costo_particion.py --config config/instances/particion-chica.yaml
Guarda en results/<case.name>/<experimento>/: <instancia>.json (y _buses.csv) apenas termina cada MILP, y resumen.json
al final. Con --solo se resuelven solo algunas instancias; el resumen usa los JSON que ya estén guardados."""

import argparse
import copy
import json
import time
from datetime import datetime

from ebus.config import load_config
from ebus.data.build_instance import build_instance
from ebus.evaluation.metrics import cota_costo
from ebus.evaluation.particion import perdida_particion, resolver_exacto
from ebus.evaluation.resultados import (
    _version_del_codigo, carpeta_de_corrida, resumen_de_corrida, tabla_de_buses,
)


def instancias_de(cfg: dict) -> dict:
    """Arma cada instancia de `particion_chica.instancias` con sus terminales y la muestra de la config. Comprueba que
    cada conjunto de `comparaciones` tenga exactamente los viajes de sus partes: si no, la comparación no vale."""
    instancias = {}
    for nombre, terminales in cfg["particion_chica"]["instancias"].items():
        cfg_i = copy.deepcopy(cfg)
        cfg_i["data"].pop("terminal", None)
        cfg_i["data"]["terminales"] = list(terminales)
        instancias[nombre] = build_instance(cfg_i)
    for conjunto, partes in cfg["particion_chica"]["comparaciones"].items():
        union = set().union(*({t.id for t in instancias[p].trips} for p in partes))
        if {t.id for t in instancias[conjunto].trips} != union:
            raise ValueError(f"{conjunto} no tiene exactamente los viajes de {partes}")
    return instancias


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--solo", nargs="*", default=None, help="resolver solo estas instancias (p. ej. --solo VN EC)")
    parser.add_argument("--out-dir", default=None, help="carpeta base de resultados (por defecto, results/)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    caso, experimento = cfg["case"]["name"], cfg["particion_chica"]["experimento"]
    carpeta = carpeta_de_corrida(caso, experimento, args.out_dir)
    carpeta.mkdir(parents=True, exist_ok=True)
    limite, gap = cfg["solver"]["time_limit_s"], cfg["solver"]["mip_gap"]

    instancias = instancias_de(cfg)
    print(f"{caso}/{experimento}: " + ", ".join(f"{n} ({len(i.trips)} viajes)" for n, i in instancias.items()), flush=True)
    for nombre, inst in instancias.items():
        if args.solo is not None and nombre not in args.solo:
            continue
        print(f"\n=== {nombre}: {len(inst.trips)} viajes, depósitos {[d.id for d in inst.depots]} "
              f"({datetime.now():%H:%M}) ===", flush=True)
        t0 = time.perf_counter()
        sol, milp = resolver_exacto(inst, limite_s=limite, mip_gap=gap)
        cota = cota_costo(inst)
        base = {
            "metodo": "milp", "instancia": nombre, "experimento": experimento,
            "terminales": cfg["particion_chica"]["instancias"][nombre],
            "muestra": cfg["data"]["muestra"], "config": args.config.replace("\\", "/"),
            "fecha": datetime.now().isoformat(timespec="seconds"), "codigo": _version_del_codigo(),
        }
        if sol is not None:
            resumen = resumen_de_corrida("milp", nombre, sol, inst, time.perf_counter() - t0, cfg, base["config"],
                                         experimento=experimento)
            tabla_de_buses(sol, inst).to_csv(carpeta / f"{nombre}_buses.csv", index=False)
        else:
            resumen = {"buses": None, "violaciones": None, "tiempo_computo_s": round(time.perf_counter() - t0, 3)}
        resumen = {**resumen, **base, "milp": milp,
                   "cota_costo": {"buses": cota.buses, "total": round(cota.total, 2)},
                   "viajes_ids": sorted(t.id for t in inst.trips)}
        (carpeta / f"{nombre}.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{nombre}: {milp['estado']}, objetivo {milp['objetivo']}, cota de Gurobi {milp['cota_gurobi']}, "
              f"buses {resumen['buses']}, violaciones {resumen['violaciones']}, {milp['tiempo_resolucion_s']:.0f} s",
              flush=True)

    guardados = {n: json.loads((carpeta / f"{n}.json").read_text(encoding="utf-8"))
                 for n in instancias if (carpeta / f"{n}.json").exists()}
    filas = {}
    for conjunto, partes in cfg["particion_chica"]["comparaciones"].items():
        if conjunto in guardados and all(p in guardados for p in partes):
            def r(n):
                return {**guardados[n]["milp"], "buses": guardados[n]["buses"]}
            filas[conjunto] = {"partes": partes, **perdida_particion([r(p) for p in partes], r(conjunto))}
    (carpeta / "resumen.json").write_text(json.dumps(
        {"experimento": experimento, "muestra": cfg["data"]["muestra"], "perdida": filas,
         "instancias": {n: {k: g["milp"][k] for k in ("estado", "objetivo", "cota_gurobi", "gap", "tiempo_resolucion_s")}
                        | {"buses": g["buses"], "viajes": len(g["viajes_ids"]), "cota_costo": g["cota_costo"]["total"]}
                        for n, g in guardados.items()}},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nCosto de la partición (separados - juntos):")
    for conjunto, f in filas.items():
        if f["costo_min"] is None:
            print(f"  {conjunto}: sin solución en algún MILP")
        elif f["exacta"]:
            print(f"  {conjunto}: {f['costo_min']:.2f} ({f['pct_min']:.2f} %), {f['buses_menos']} buses menos juntos (exacto)")
        else:
            print(f"  {conjunto}: entre {f['costo_min']:.2f} y {f['costo_max']:.2f} ({f['pct_min']:.2f} % a "
                  f"{f['pct_max']:.2f} %), {f['buses_menos']} buses menos juntos (algún MILP sin óptimo)")
    print(f"Resultados en {carpeta}")


if __name__ == "__main__":
    main()
