"""Matriz de deadheads d_ij entre paraderos y depósitos (Paso II): lectura y estimación."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ebus.model.entities import Station

REGLA_MINIMO = "minimo"
REGLA_MEDIANA = "mediana"


def load_deadhead_matrix(
    path: str | Path,
) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], float]]:
    """Lee la matriz de deadheads (ver docs/datos/datos_requeridos.md) y devuelve
    (`deadhead_time_min`, `deadhead_distance_km`) por par `(from_id, to_id)`, como los espera
    `Instance.deadhead_min` y `Instance.deadhead_km`. Las columnas `energy_kwh` y `cost_usd` se
    ignoran: la energía y el costo se calculan desde la config."""
    df = pd.read_csv(path, dtype={"from_id": str, "to_id": str})
    faltan = {"from_id", "to_id", "deadhead_time_min", "deadhead_distance_km"} - set(df.columns)
    if faltan:
        raise ValueError(f"{path}: faltan las columnas {sorted(faltan)}")
    if not (df["deadhead_time_min"] % 1 == 0).all():
        raise ValueError(f"{path}: deadhead_time_min debe ser entero (minutos enteros)")
    if df.duplicated(["from_id", "to_id"]).any():
        raise ValueError(f"{path}: hay pares (from_id, to_id) repetidos")
    keys = list(zip(df["from_id"], df["to_id"]))
    minutes = {k: int(v) for k, v in zip(keys, df["deadhead_time_min"])}
    km = {k: float(v) for k, v in zip(keys, df["deadhead_distance_km"])}
    return minutes, km


def pares_faltantes(minutos: dict[tuple[str, str], int], nodos: set[str]) -> list[tuple[str, str]]:
    """Pares ordenados `(i, j)`, `i != j`, de `nodos` que no están en la matriz (orden alfabético).
    Una matriz completa y dirigida devuelve `[]`."""
    return [(i, j) for i in sorted(nodos) for j in sorted(nodos) if i != j and (i, j) not in minutos]


def estimar_matriz_deadheads(
    viajes: pd.DataFrame,
    depositos: list[Station],
    velocidad_kmh: float,
    factor_desvio: float,
    kwh_por_km: float,
    costo_por_km: float,
    regla_km_viaje: str = REGLA_MINIMO,
) -> pd.DataFrame:
    """Matriz de deadheads completa y dirigida entre los paraderos de `viajes` y los `depositos` (incluidos los pares
    depósito ↔ depósito, con `tipo = deadhead`: un bus puede salir de uno y cargar en otro), con el método de
    docs/decisiones/datos.md ("Matriz de deadheads de VN-10"). Reemplazar por ruteo en red vial (OSM) si hay tiempo.

    - `km`: haversine × `factor_desvio`, corregido con los `km` de los viajes de `viajes` que tienen el mismo par
      (origen, destino), según `regla_km_viaje` (`deadhead_estimation.trip_km_rule` en la config):
      - `"minimo"` (por defecto): el menor entre la estimación y los `km` de esos viajes. Un bus siempre puede seguir el
        recorrido de un viaje, así que esos km son una cota, no una obligación; en recorridos casi circulares (origen y
        destino a metros, viaje de 20 km) el deadhead no da la vuelta (docs/decisiones/instancias.md, "Día completo de un
        electroterminal").
      - `"mediana"`: la mediana de los `km` de esos viajes. Es la regla con que se generó `deadhead_VN-10.csv`.
    - `minutos`: `km / velocidad_kmh × 60`, redondeado, mínimo 1.
    - `energy_kwh` = `km × kwh_por_km` y `cost_usd` = `km × costo_por_km` (informativas: `load_deadhead_matrix` las
      ignora).

    `viajes` trae las columnas de `viajes_dia_habil.csv` (`stop_origen`, `lat_origen`, `lon_origen`, `stop_destino`,
    `lat_destino`, `lon_destino`, `km`). Devuelve las columnas del diccionario, ordenadas por (`from_id`, `to_id`)."""
    if viajes.empty:
        raise ValueError("No hay viajes con que estimar la matriz de deadheads")
    if velocidad_kmh <= 0 or factor_desvio <= 0:
        raise ValueError(f"velocidad_kmh y factor_desvio deben ser positivos ({velocidad_kmh}, {factor_desvio})")
    if regla_km_viaje not in (REGLA_MINIMO, REGLA_MEDIANA):
        raise ValueError(f"regla_km_viaje desconocida: {regla_km_viaje!r} (use {REGLA_MINIMO!r} o {REGLA_MEDIANA!r})")
    if not depositos:
        raise ValueError("Hace falta al menos un depósito para la matriz de deadheads")
    nodos: dict[str, Station] = {d.id: d for d in depositos}
    ids_depositos = set(nodos)
    for r in viajes.itertuples():
        nodos.setdefault(r.stop_origen, Station(r.stop_origen, float(r.lat_origen), float(r.lon_origen)))
        nodos.setdefault(r.stop_destino, Station(r.stop_destino, float(r.lat_destino), float(r.lon_destino)))
    por_par = viajes.groupby(["stop_origen", "stop_destino"])["km"]
    km_viaje = (por_par.min() if regla_km_viaje == REGLA_MINIMO else por_par.median()).to_dict()

    filas = []
    for i in sorted(nodos):
        for j in sorted(nodos):
            if i == j:
                continue
            km = nodos[i].distance_km_to(nodos[j]) * factor_desvio
            if (i, j) in km_viaje:
                km = min(km, km_viaje[i, j]) if regla_km_viaje == REGLA_MINIMO else km_viaje[i, j]
            km = float(km)
            if i in ids_depositos and j not in ids_depositos:
                tipo = "pullout"
            elif j in ids_depositos and i not in ids_depositos:
                tipo = "pullin"
            else:
                tipo = "deadhead"
            filas.append((i, j, tipo, round(km, 3), max(1, round(km / velocidad_kmh * 60)),
                          round(km * kwh_por_km, 3), round(km * costo_por_km, 3)))
    return pd.DataFrame(filas, columns=["from_id", "to_id", "tipo", "deadhead_distance_km", "deadhead_time_min",
                                        "energy_kwh", "cost_usd"])
