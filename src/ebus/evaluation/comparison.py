"""Tabla comparativa entre métodos de solución (tipo Tablas 7-8 del paper): baseline vs. MILP
vs. ALNS para la misma instancia. No le importa qué método generó cada `Solution`."""

from __future__ import annotations

import json
import numbers
from pathlib import Path

import pandas as pd

from ebus.evaluation.feasibility import check_solution
from ebus.model.cost import evaluate
from ebus.model.entities import Instance, Solution


def compare_methods(
    solutions: dict[str, Solution],
    inst: Instance,
    computation_time_s: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Una fila por método: costo total y su desglose, nº de buses, nº de violaciones de
    factibilidad, tiempo de cómputo (si se entrega) y brecha % respecto al método más caro
    (típicamente el baseline, que sirve de cota superior de referencia)."""
    computation_time_s = computation_time_s or {}
    rows = []
    for name, sol in solutions.items():
        breakdown = evaluate(sol, inst)
        violations = check_solution(sol, inst)
        rows.append(
            {
                "metodo": name,
                "costo_total": breakdown.total,
                "costo_fijo_buses": breakdown.bus_fixed_cost,
                "deadhead": breakdown.deadhead,
                "carga_diurna": breakdown.daytime_charging,
                "carga_nocturna": breakdown.overnight_charging,
                "n_buses": sol.n_buses,
                "violaciones": len(violations),
                "tiempo_computo_s": computation_time_s.get(name),
            }
        )

    df = pd.DataFrame(rows).set_index("metodo")
    peor_costo = df["costo_total"].max()
    df["brecha_vs_peor_%"] = (peor_costo - df["costo_total"]) / peor_costo * 100
    return df


def tabla_de_resultados(instancia: str, carpeta: str | Path | None = None,
                        experimento: str | None = None) -> pd.DataFrame:
    """Tabla comparativa armada con los resúmenes guardados en `<carpeta>/<instancia>/` y sus subcarpetas (ver
    `resultados.py`), sin volver a correr nada. Una fila por corrida, ordenadas por costo, con la mejora porcentual
    respecto del baseline (si hay uno). El índice identifica la corrida por su ruta dentro de la carpeta de la
    instancia: `baseline`, `E2-dmax/d20_s42`. Con `experimento`, solo las corridas de ese experimento y el baseline."""
    raiz = (Path(carpeta) if carpeta is not None else Path(__file__).resolve().parents[3] / "results") / instancia
    filas = []
    for ruta in sorted(raiz.rglob("*.json")) if raiz.is_dir() else []:
        resumen = json.loads(ruta.read_text(encoding="utf-8"))
        if resumen.get("instancia") != instancia:
            continue
        nombre = ruta.relative_to(raiz).with_suffix("").as_posix()
        exp = ruta.parent.relative_to(raiz).as_posix()
        exp = "" if exp == "." else exp
        if experimento is not None and resumen["metodo"] != "baseline" and not (exp + "/").startswith(experimento + "/"):
            continue
        corrida = resumen.get("corrida") or {}
        commit = (resumen.get("codigo") or {}).get("commit")
        filas.append({
            "corrida": nombre,
            "metodo": resumen["metodo"],
            "experimento": exp,
            "costo_total": resumen["costo"]["total"],
            "costo_fijo_buses": resumen["costo"]["costo_fijo_buses"],
            "deadhead": resumen["costo"]["deadhead"],
            "carga_diurna": resumen["costo"]["carga_diurna"],
            "carga_nocturna": resumen["costo"]["carga_nocturna"],
            "buses": resumen["buses"],
            "cota_inferior_buses": resumen["cota_inferior_buses"]["viajes_simultaneos"],
            "violaciones": len(resumen["violaciones"]),
            "tiempo_computo_s": resumen["tiempo_computo_s"],
            "iteraciones": corrida.get("iteraciones"),
            "semilla": corrida.get("semilla"),
            "destroy_max": corrida.get("destroy_max"),
            "accept_probability": corrida.get("accept_probability"),
            "config": resumen.get("config"),
            "fecha": resumen.get("fecha"),
            "commit": commit[:7] if commit else None,
        })
    if not filas:
        raise FileNotFoundError(f"No hay resultados de la instancia {instancia} en {raiz}")
    df = pd.DataFrame(filas).set_index("corrida").sort_values("costo_total", kind="stable")
    if "baseline" in df.index:
        base = df.loc["baseline", "costo_total"]
        df["mejora_vs_baseline_%"] = ((base - df["costo_total"]) / base * 100).round(3)
    return df


INICIO_REGISTRO = "<!-- corridas:inicio (lo escribe scripts/comparar.py; no editar a mano) -->"
FIN_REGISTRO = "<!-- corridas:fin -->"


def tabla_markdown(tabla: pd.DataFrame) -> str:
    """Las corridas de `tabla_de_resultados` como tabla Markdown, para el registro de la instancia."""
    def valor(v) -> str:
        """Números como en los docs: miles separados por espacio y coma decimal (229 326,95)."""
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        if isinstance(v, numbers.Number) and not isinstance(v, bool):
            if float(v).is_integer():
                texto = f"{int(v):,}"
            elif abs(v) >= 100:
                texto = f"{v:,.2f}"
            else:
                texto = f"{v:.4f}".rstrip("0").rstrip(".")
            return texto.replace(",", " ").replace(".", ",")
        return str(v)

    columnas = [("corrida", "Corrida"), ("fecha", "Fecha"), ("commit", "Commit"), ("iteraciones", "Iteraciones"),
                ("semilla", "Semilla"), ("destroy_max", "D_max"), ("accept_probability", "Pr_accept"),
                ("buses", "Buses"), ("costo_total", "Costo"), ("mejora_vs_baseline_%", "Mejora vs. baseline (%)"),
                ("tiempo_computo_s", "Tiempo (s)")]
    columnas = [(c, n) for c, n in columnas if c == "corrida" or c in tabla.columns]
    filas = ["| " + " | ".join(n for _, n in columnas) + " |", "|" + "---|" * len(columnas)]
    orden = sorted(tabla.index, key=lambda n: ("/" in n, n))
    for nombre, fila in tabla.loc[orden].iterrows():
        celdas = [f"`{nombre}`"] + [valor(fila[c]) for c, _ in columnas[1:]]
        celdas[1] = celdas[1][:10]
        filas.append("| " + " | ".join(celdas) + " |")
    return "\n".join(filas)


def actualizar_registro(ruta: str | Path, instancia: str, tabla: pd.DataFrame) -> Path:
    """Reescribe la tabla de corridas del registro de la instancia (`results/<instancia>/README.md`), entre las marcas
    `INICIO_REGISTRO` y `FIN_REGISTRO`, sin tocar el resto (la descripción de los experimentos, que se escribe a mano).
    Si el archivo no existe, lo crea con una sección de experimentos vacía."""
    ruta = Path(ruta)
    bloque = f"{INICIO_REGISTRO}\n{tabla_markdown(tabla)}\n{FIN_REGISTRO}"
    if ruta.exists():
        texto = ruta.read_text(encoding="utf-8")
        if INICIO_REGISTRO in texto and FIN_REGISTRO in texto:
            antes, resto = texto.split(INICIO_REGISTRO, 1)
            texto = antes + bloque + resto.split(FIN_REGISTRO, 1)[1]
        else:
            texto = texto.rstrip("\n") + "\n\n## Corridas\n\n" + bloque + "\n"
    else:
        texto = (f"# Resultados: {instancia}\n\n## Experimentos\n\n| ID | Pregunta | Configs | Conclusión |\n"
                 f"|---|---|---|---|\n\n## Corridas\n\n{bloque}\n")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(texto, encoding="utf-8")
    return ruta
