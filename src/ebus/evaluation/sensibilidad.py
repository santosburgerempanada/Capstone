"""Análisis de sensibilidad del ALNS (análogo a la Fig. 8 del paper): cambia un parámetro a la vez, deja los demás como
en la config, y corre el ALNS con varias semillas desde la misma solución inicial."""

from __future__ import annotations

import copy

import pandas as pd

from ebus.heuristics.alns import run_alns
from ebus.model.entities import Instance, Solution


def sensibilidad(inicial: Solution, inst: Instance, alns_cfg: dict, variaciones: dict[str, list],
                 iteraciones: int, semillas: list[int]) -> pd.DataFrame:
    """Una fila por (parámetro, valor, semilla): costo y buses de la mejor solución y tiempo. `variaciones` mapea
    nombres de la sección `alns` de la config (p. ej. `destroy_max`) a los valores a probar."""
    filas = []
    for parametro, valores in variaciones.items():
        if parametro not in alns_cfg:
            raise KeyError(f"{parametro} no es un parámetro de la sección alns de la config")
        for valor in valores:
            cfg = copy.deepcopy(alns_cfg)
            cfg[parametro] = valor
            for semilla in semillas:
                resultado = run_alns(inicial, inst, cfg, iteraciones=iteraciones, semilla=semilla)
                filas.append({
                    "parametro": parametro, "valor": valor, "semilla": semilla, "iteraciones": iteraciones,
                    "costo": round(resultado.mejor.cost, 2), "buses": resultado.mejor.n_buses,
                    "costo_inicial": round(resultado.costo_inicial, 2), "tiempo_s": round(resultado.tiempo_s, 2),
                })
    return pd.DataFrame(filas)


def resumen_sensibilidad(tabla: pd.DataFrame) -> pd.DataFrame:
    """Promedio, mínimo y máximo del costo (y buses promedio) por parámetro y valor."""
    return (tabla.groupby(["parametro", "valor"])
            .agg(costo_promedio=("costo", "mean"), costo_min=("costo", "min"), costo_max=("costo", "max"),
                 buses_promedio=("buses", "mean"), tiempo_promedio_s=("tiempo_s", "mean"))
            .round(2))
