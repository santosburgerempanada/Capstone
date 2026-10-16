"""Guarda en disco el resultado de una corrida (baseline, ALNS o MILP) para poder compararlo después sin
volver a ejecutarla. Organización de `results/` (ver docs/decisiones/resultados.md):

    results/<instancia>/baseline.json                   el caso base (determinista: uno por instancia)
    results/<instancia>/<experimento>/<nombre>.json     una corrida de un experimento (p. ej. E2-dmax/d20_s42)

Cada corrida escribe `<nombre>.json` (resumen: costo y desglose, buses, violaciones, parámetros, código usado),
`<nombre>_buses.csv` (una fila por bus) y, en el ALNS, `<nombre>_historia.csv.gz` (una fila por iteración). Una
corrida con el mismo nombre sobrescribe la anterior: por eso cada experimento tiene su propio identificador."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from ebus.evaluation.feasibility import check_solution
from ebus.evaluation.metrics import depot_occupancy_profile, minutes_over_capacity, peak_concurrent_trips
from ebus.model.cost import chain_deadhead_km, evaluate, soc_at_return
from ebus.model.entities import Instance, Solution, all_recharge_events

RAIZ = Path(__file__).resolve().parents[3]


def _hhmm(minuto: int) -> str:
    return f"{minuto // 60:02d}:{minuto % 60:02d}"


def _version_del_codigo() -> dict:
    """Commit con que se corrió y si había cambios sin commitear fuera de `results/` (None si git no está
    disponible)."""
    def git(*args: str) -> str | None:
        try:
            r = subprocess.run(["git", *args], cwd=RAIZ, capture_output=True, text=True, timeout=10)
            return r.stdout.strip() if r.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    commit, cambios = git("rev-parse", "HEAD"), git("status", "--porcelain", "--", ".", ":(exclude)results")
    return {"commit": commit, "cambios_sin_commitear": None if cambios is None else bool(cambios)}


def carpeta_de_corrida(instancia: str, experimento: str | None = None, out_dir: str | Path | None = None) -> Path:
    """`results/<instancia>/` (o `<out_dir>/<instancia>/`) y, si la corrida es de un experimento, su subcarpeta."""
    carpeta = (Path(out_dir) if out_dir is not None else RAIZ / "results") / instancia
    return carpeta / experimento if experimento else carpeta


def variante_por_defecto(config_path: str | Path, instancia: str) -> str:
    """Nombre de la variante a partir de la config: `VN-dia-d20.yaml` de la instancia `VN-dia` es `d20`, y la config
    base de la instancia (`VN-dia.yaml`) es `base`. Una config con otro nombre se usa entera."""
    nombre = Path(config_path).stem
    if nombre == instancia:
        return "base"
    return nombre[len(instancia) + 1:] if nombre.startswith(instancia + "-") else nombre


def tabla_de_buses(sol: Solution, inst: Instance) -> pd.DataFrame:
    """Una fila por bus: depósitos, viajes, horarios, SOC (antes del primer viaje, tras el último y al llegar
    al depósito), km de deadhead y número de cargas."""
    filas = []
    for chain in sol.chains:
        primero, ultimo = chain.trips[0], chain.trips[-1]
        filas.append({
            "bus_id": chain.bus_id,
            "deposito_salida": chain.depot_id,
            "deposito_retorno": chain.return_depot_id,
            "n_viajes": len(chain.trips),
            "viajes": ";".join(t.id for t in chain.trips),
            "primera_salida": _hhmm(primero.start_time),
            "ultima_llegada": _hhmm(ultimo.end_time),
            "soc_antes_primer_viaje": round(chain.soc_before[0], 4),
            "soc_tras_ultimo_viaje": round(chain.soc_after[-1], 4),
            "soc_al_volver": round(soc_at_return(chain, inst), 4),
            "deadhead_km": round(chain_deadhead_km(chain, inst), 3),
            "n_cargas": sum(1 for e in all_recharge_events(chain) if e.power_kw > 0),
        })
    return pd.DataFrame(filas)


def resumen_de_corrida(
    metodo: str, instancia: str, sol: Solution, inst: Instance, tiempo_s: float | None = None,
    cfg: dict | None = None, config_path: str | None = None, corrida: dict | None = None,
    experimento: str | None = None,
) -> dict:
    """Resumen de una corrida, listo para serializar a JSON. `corrida` guarda datos propios del método (p. ej., las
    iteraciones y la semilla del ALNS) para poder compararlos después."""
    costo = evaluate(sol, inst)
    violaciones = check_solution(sol, inst)
    cota, minuto_cota = peak_concurrent_trips(inst)
    por_bus: dict[str, int] = {}
    for chain in sol.chains:
        por_bus[str(len(chain.trips))] = por_bus.get(str(len(chain.trips)), 0) + 1

    depositos = []
    for depot in inst.depots:
        perfil = depot_occupancy_profile(sol, inst, depot.id)
        depositos.append({
            "id": depot.id, "plazas": depot.capacity, "cargadores": depot.n_chargers, "potencia_kw": depot.power_kw,
            "minutos_con_mas_buses_que_plazas": None if depot.capacity is None else minutes_over_capacity(perfil, depot.capacity),
            "minutos_del_horizonte": len(perfil),
        })

    bus = inst.bus_type
    return {
        "metodo": metodo,
        "instancia": instancia,
        "experimento": experimento,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "corrida": corrida or {},
        "config": config_path,
        "codigo": _version_del_codigo(),
        "tiempo_computo_s": None if tiempo_s is None else round(tiempo_s, 3),
        "moneda": (cfg or {}).get("costs", {}).get("currency"),
        "costo": {
            "total": round(costo.total, 2),
            "costo_fijo_buses": round(costo.bus_fixed_cost, 2),
            "deadhead": round(costo.deadhead, 2),
            "carga_diurna": round(costo.daytime_charging, 2),
            "carga_nocturna": round(costo.overnight_charging, 2),
        },
        "buses": sol.n_buses,
        "cota_inferior_buses": {"viajes_simultaneos": cota, "hora": _hhmm(minuto_cota)},
        "viajes": {
            "total": len(inst.trips),
            "cubiertos": sum(len(c.trips) for c in sol.chains),
            "sin_asignar": len(sol.unassigned),
        },
        "viajes_por_bus": dict(sorted(por_bus.items(), key=lambda kv: int(kv[0]))),
        "cargas": sum(1 for c in sol.chains for e in all_recharge_events(c) if e.power_kw > 0),
        "depositos": depositos,
        "violaciones": violaciones,
        "parametros": {
            "horizonte": [_hhmm(inst.horizon_start), _hhmm(inst.horizon_end)],
            "soc_min": inst.soc_min, "soc_inicial": inst.soc_initial, "soc_max": inst.soc_max,
            "bateria_kwh": bus.battery_kwh, "consumo_viaje_kwh_km": bus.trip_kwh_per_km,
            "consumo_deadhead_kwh_km": round(bus.deadhead_kwh_per_km, 4),
            "costo_por_bus": bus.fixed_cost_per_bus, "costo_deadhead_por_km": inst.deadhead_cost_per_km,
            "precio_nocturno_kwh": inst.tariff.overnight_price_per_kwh,
        },
    }


def guardar_corrida(
    metodo: str, instancia: str, sol: Solution, inst: Instance, tiempo_s: float | None = None,
    out_dir: str | Path | None = None, cfg: dict | None = None, config_path: str | None = None,
    corrida: dict | None = None, experimento: str | None = None, nombre: str | None = None,
) -> tuple[Path, Path]:
    """Escribe el resumen (JSON) y la tabla de buses (CSV) de la corrida en `carpeta_de_corrida(...)`, con el nombre
    `nombre` (por defecto, el del método). Devuelve ambas rutas."""
    carpeta = carpeta_de_corrida(instancia, experimento, out_dir)
    carpeta.mkdir(parents=True, exist_ok=True)
    base = nombre or metodo
    ruta_json, ruta_csv = carpeta / f"{base}.json", carpeta / f"{base}_buses.csv"

    resumen = resumen_de_corrida(metodo, instancia, sol, inst, tiempo_s, cfg, config_path, corrida, experimento)
    ruta_json.write_text(json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tabla_de_buses(sol, inst).to_csv(ruta_csv, index=False)
    return ruta_json, ruta_csv


def guardar_historia(nombre: str, instancia: str, historia: list[dict], out_dir: str | Path | None = None,
                     experimento: str | None = None) -> Path:
    """Escribe la historia de una corrida (una fila por iteración: operadores, resultado, costos y temperatura) en
    `<nombre>_historia.csv.gz`, junto a su JSON, para graficar la convergencia. Va comprimida (15 000 iteraciones son
    ~1,5 MB sin comprimir) y sin fecha en el gzip, para que la misma historia dé el mismo archivo;
    `pd.read_csv` la lee directamente."""
    carpeta = carpeta_de_corrida(instancia, experimento, out_dir)
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"{nombre}_historia.csv.gz"
    pd.DataFrame(historia).to_csv(ruta, index=False, compression={"method": "gzip", "mtime": 0})
    return ruta
