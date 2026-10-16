"""Construye la `Instance` de un subproblema de la partición (un electroterminal y un bloque
horario, o el día completo de un electroterminal) a partir de los CSV de `data/`. Ver docs/decisiones/instancias.md,
docs/planes/plan_de_accion.md (Fase 2) y docs/planes/cerrados/plan_dia_completo_vn.md."""

from __future__ import annotations

import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ebus.data.deadhead import load_deadhead_matrix, pares_faltantes
from ebus.data.depots import depot_ids_por_nombre, load_depots
from ebus.data.fleet import load_bus_types
from ebus.data.horizonte import calcular_horizonte, margen_horizonte
from ebus.data.parameters import load_parameters
from ebus.data.tariffs import load_santiago_tariff
from ebus.data.tiempos import hhmm_a_minutos, llegada_a_minutos, salida_a_minutos
from ebus.model.entities import Depot, Instance, Station, Trip

RAIZ = Path(__file__).resolve().parents[3]

_COLUMNAS_VIAJES = [
    "viaje_id", "terminal", "bloque", "bloque_ini", "route_id", "shape_id", "dep_s", "arr_s", "km",
    "stop_origen", "lat_origen", "lon_origen", "stop_destino", "lat_destino", "lon_destino",
]
_SEGUNDOS_DIA = 24 * 3600


def _ruta(texto: str | Path) -> Path:
    """Las rutas relativas de la config se resuelven desde la raíz del proyecto."""
    ruta = Path(texto)
    return ruta if ruta.is_absolute() else RAIZ / ruta


def _requerido(seccion: dict, clave: str, donde: str):
    valor = seccion.get(clave)
    if valor is None:
        raise ValueError(f"Falta {donde}.{clave} en la config del subproblema")
    return valor


def leer_viajes(cfg: dict) -> tuple[pd.DataFrame, list[str], int]:
    """Viajes de la instancia, desde `data.trips_file`. Devuelve (viajes, terminales, inicio en minutos).

    La instancia se elige con una de tres claves de `data` (exactamente una):
    - `block_id`: un bloque de la partición (p. ej. VN-10). El inicio es el de ese bloque (`bloque_ini`).
    - `terminal`: todos los viajes del día de un electroterminal (p. ej. "Vespucio Norte").
    - `terminales`: todos los viajes del día de varios electroterminales, con sus depósitos (p. ej. Vespucio Norte y El
      Conquistador; ver docs/planes/cerrados/plan_dos_terminales.md). Cualquier bus puede hacer cualquier viaje: el terminal que la
      partición asignó a cada viaje queda solo como dato.

    Con `terminal` o `terminales` el día empieza en `network.day_start`: los viajes que salen antes se corren 24 h y
    pasan al final del día (ver docs/decisiones/instancias.md, "Día completo de un electroterminal"). En `dep_s` y `arr_s` queda la
    hora ya corrida. Con `data.muestra` se toma una muestra al azar de los viajes de cada terminal (`_muestra_por_terminal`)."""
    data, network = cfg["data"], cfg["network"]
    bloque, terminal, terminales = data.get("block_id"), data.get("terminal"), data.get("terminales")
    if sum(clave is not None for clave in (bloque, terminal, terminales)) != 1:
        raise ValueError("La config del subproblema debe fijar exactamente una de data.block_id, data.terminal o "
                         "data.terminales")

    df = pd.read_csv(
        _ruta(_requerido(data, "trips_file", "data")), usecols=_COLUMNAS_VIAJES,
        dtype={"stop_origen": str, "stop_destino": str, "route_id": str, "shape_id": str},
    )
    if bloque is not None:
        df = df[df["bloque"] == bloque]
        if df.empty:
            raise ValueError(f"El bloque {bloque!r} no tiene viajes en {data['trips_file']}")
        terminales = df["terminal"].unique()
        if len(terminales) != 1:
            raise ValueError(f"El bloque {bloque!r} debería tener un solo terminal; tiene {list(terminales)}")
        return df, [str(terminales[0])], hhmm_a_minutos(df["bloque_ini"].iloc[0])

    nombres = [str(terminal)] if terminal is not None else [str(t) for t in terminales]
    if not nombres or len(set(nombres)) != len(nombres):
        raise ValueError(f"data.terminales debe tener terminales distintos; tiene {nombres}")
    df = df[df["terminal"].isin(nombres)].copy()
    sin_viajes = [t for t in nombres if t not in set(df["terminal"])]
    if sin_viajes:
        raise ValueError(f"El terminal {sin_viajes[0]!r} no tiene viajes en {data['trips_file']}")
    if data.get("muestra") is not None:
        df = _muestra_por_terminal(df, nombres, data["muestra"])
    inicio_dia = hhmm_a_minutos(_requerido(network, "day_start", "network"))
    madrugada = df["dep_s"] < inicio_dia * 60
    df.loc[madrugada, ["dep_s", "arr_s"]] += _SEGUNDOS_DIA
    return df, nombres, inicio_dia


def _muestra_por_terminal(df: pd.DataFrame, nombres: list[str], muestra: dict) -> pd.DataFrame:
    """`muestra.viajes_por_terminal` viajes al azar de cada terminal (todos, si tiene menos), con `muestra.semilla`, para las
    instancias chicas del costo de la partición (sección 4b de docs/planes/entrega_2.md). La muestra de un terminal depende
    solo de su nombre, del tamaño y de la semilla, no de los otros terminales de la instancia: la de varios terminales es
    exactamente la unión de las de cada uno."""
    por_terminal, semilla = muestra.get("viajes_por_terminal"), muestra.get("semilla")
    if not isinstance(por_terminal, int) or por_terminal <= 0 or not isinstance(semilla, int):
        raise ValueError(f"data.muestra necesita viajes_por_terminal (entero positivo) y semilla (entero); tiene {muestra}")
    partes = []
    for nombre in nombres:
        viajes = df[df["terminal"] == nombre].sort_values("viaje_id")
        rng = np.random.default_rng([semilla, zlib.crc32(nombre.encode("utf-8"))])
        elegidos = rng.choice(len(viajes), size=min(por_terminal, len(viajes)), replace=False)
        partes.append(viajes.iloc[np.sort(elegidos)])
    return pd.concat(partes)


def depositos_de(cfg: dict, terminales: list[str]) -> list[Depot]:
    """Depósitos de los `terminales` (el electroterminal es a la vez depósito y lugar de carga), con sus cargadores y
    potencia, en el orden de `terminales`.

    La capacidad de estacionamiento de cada depósito (buses dentro a la vez, también de noche) sale de la config, no de
    `depots.csv` (docs/decisiones/modelo.md). Es propia de cada terminal: `depots.capacidad_estacionamiento` para todos y
    las excepciones de `depots.capacidad_por_deposito` (por nombre de terminal), según la regla máx(1 000, flota del
    baseline del terminal solo) + 200. Sin ninguna de las dos, la capacidad queda sin límite (`None`)."""
    data = cfg["data"]
    depots_file = _ruta(_requerido(data, "depots_file", "data"))
    ids = depot_ids_por_nombre(depots_file)
    for terminal in terminales:
        if terminal not in ids:
            raise ValueError(f"El terminal {terminal!r} no está en {depots_file.name}: {sorted(ids)}")
    por_id = {d.id: d for d in load_depots(depots_file, _ruta(_requerido(data, "charger_capacity_file", "data")))}
    seccion = cfg.get("depots") or {}
    general = seccion.get("capacidad_estacionamiento")
    excepciones = seccion.get("capacidad_por_deposito") or {}
    desconocidos = sorted(set(excepciones) - set(ids))
    if desconocidos:
        raise ValueError(f"depots.capacidad_por_deposito nombra terminales que no están en {depots_file.name}: "
                         f"{desconocidos}")
    for nombre, valor in {"capacidad_estacionamiento": general, **excepciones}.items():
        if valor is not None and (not isinstance(valor, int) or valor < 1):
            raise ValueError(f"La capacidad de estacionamiento ({nombre}) debe ser un entero positivo: {valor!r}")
    return [replace(por_id[ids[t]], capacity=excepciones.get(t, general)) for t in terminales]


def construir_instancia_subproblema(cfg: dict) -> Instance:
    """Arma la `Instance` de un bloque (`data.block_id`), del día completo de un terminal (`data.terminal`) o de varios
    terminales con sus depósitos (`data.terminales`; ver `leer_viajes`) leyendo los archivos que indica `data` en la
    config.

    Los viajes llevan horas en minutos enteros (salida hacia abajo, llegada hacia arriba). El horizonte se extiende
    por el margen de deadhead (`network.horizon_margin`) y los buses parten con `battery.soc_initial`."""
    data, battery, costs, network = cfg["data"], cfg["battery"], cfg["costs"], cfg["network"]

    df, terminales, inicio = leer_viajes(cfg)
    trips = [
        Trip(
            id=r.viaje_id, route_id=r.route_id, start_station=r.stop_origen, end_station=r.stop_destino,
            start_time=salida_a_minutos(r.dep_s), end_time=llegada_a_minutos(r.arr_s),
            distance_km=float(r.km), shape_id=r.shape_id, terminal=r.terminal, block_id=r.bloque,
        )
        for r in df.itertuples()
    ]
    station_coords: dict[str, Station] = {}
    for r in df.itertuples():
        station_coords.setdefault(r.stop_origen, Station(r.stop_origen, float(r.lat_origen), float(r.lon_origen)))
        station_coords.setdefault(r.stop_destino, Station(r.stop_destino, float(r.lat_destino), float(r.lon_destino)))

    depots = depositos_de(cfg, terminales)

    params = load_parameters(_ruta(_requerido(data, "parameters_file", "data")))
    buses = load_bus_types(
        _ruta(_requerido(data, "fleet_file", "data")),
        consumption_kwh_per_km=battery["consumption_kwh_per_km"],
        deadhead_consumption_factor=battery["deadhead_consumption_factor"],
        fixed_cost_per_bus=params["vehicle_fixed_cost"],
    )
    tipo = data.get("bus_type_id")
    if tipo is not None:
        buses = [b for b in buses if b.id == tipo]
    if len(buses) != 1:
        raise ValueError("Se espera un solo tipo de bus por instancia; fije data.bus_type_id "
                         f"(hay {[b.id for b in buses]})")
    tariff = load_santiago_tariff(
        _ruta(_requerido(data, "tariffs_file", "data")), _requerido(costs, "overnight_price_per_kwh", "costs")
    )

    minutos, km = load_deadhead_matrix(_ruta(_requerido(data, "deadhead_file", "data")))
    faltan = pares_faltantes(minutos, set(station_coords) | {d.id for d in depots})
    if faltan:
        raise ValueError(f"A la matriz de deadheads le faltan {len(faltan)} pares, p. ej. {faltan[:3]}")
    horizon_start, horizon_end = calcular_horizonte(
        trips, inicio, margen_horizonte(minutos, _requerido(network, "horizon_margin", "network"))
    )

    return Instance(
        trips=trips, depots=depots, bus_type=buses[0], tariff=tariff,
        deadhead_min=minutos, deadhead_km=km, deadhead_cost_per_km=_requerido(costs, "deadhead_cost_per_km", "costs"),
        horizon_start=horizon_start, horizon_end=horizon_end, time_step=network["time_step_min"],
        soc_min=battery["soc_min"], soc_max=battery["soc_max"], soc_initial=battery.get("soc_initial"),
        station_coords=station_coords,
    )
