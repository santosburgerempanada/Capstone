"""Partición del problema: asigna cada recorrido a un electroterminal (el más cercano al
centroide del trazado) y divide los viajes de cada terminal en bloques horarios.

Entrada: GTFS en `data/raw/` + parámetros en `config/particion.yaml`.
Salida: CSV (y un Excel piloto) en `data/processed/<particion.nombre>/`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def sec(h) -> int:
    """'HH:MM:SS' -> segundos desde las 00:00."""
    a = str(h).strip().split(":")
    return int(a[0]) * 3600 + int(a[1]) * 60 + int(a[2])


def hhmm(s) -> str:
    s = int(round(s))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}"


def cargar(cfg: dict):
    """Lee los insumos indicados en `cfg['data']`. El primer vehículo de `fleet_file` es el
    bus del subproblema."""
    d = cfg["data"]
    gtfs = Path(d["gtfs_dir"])
    trips = pd.read_csv(gtfs / "trips.txt", dtype=str)
    freqs = pd.read_csv(gtfs / "frequencies.txt", dtype={"trip_id": str, "headway_secs": int})
    stops = pd.read_csv(gtfs / "stops.txt", dtype={"stop_id": str},
                        usecols=["stop_id", "stop_lat", "stop_lon"])
    st = pd.read_csv(gtfs / "stop_times.txt", dtype=str,
                     usecols=["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"])
    dist = pd.read_csv(d["shape_distances_file"])
    depots = pd.read_csv(d["depots_file"])
    veh = pd.read_csv(d["fleet_file"]).iloc[0]
    return gtfs, trips, freqs, stops, st, dist, depots, veh


def asignar_centroide(gtfs_dir, trips_servicio, depots, nivel="ruta") -> pd.DataFrame:
    """Terminal más cercano al centroide (ponderado por largo) del trazado."""
    sh = pd.read_csv(Path(gtfs_dir) / "shapes.txt", dtype={"shape_id": str})

    sh = sh[sh.shape_id.isin(trips_servicio.shape_id.unique())]
    sh = sh.sort_values(["shape_id", "shape_pt_sequence"])
    lat0 = sh.shape_pt_lat.mean()
    kx, ky = 111.32 * np.cos(np.radians(lat0)), 110.57
    sh["x"] = sh.shape_pt_lon * kx
    sh["y"] = sh.shape_pt_lat * ky
    nxt = sh.groupby("shape_id")[["x", "y"]].shift(-1)
    seg = sh.assign(x2=nxt.x, y2=nxt.y).dropna(subset=["x2"])
    seg["len"] = np.hypot(seg.x2 - seg.x, seg.y2 - seg.y)
    seg["mx"] = (seg.x + seg.x2) / 2 * seg.len
    seg["my"] = (seg.y + seg.y2) / 2 * seg.len
    seg = seg.merge(trips_servicio[["shape_id", "route_id"]].drop_duplicates(), on="shape_id")
    clave = "route_id" if nivel == "ruta" else "shape_id"
    c = seg.groupby(clave).agg(L=("len", "sum"), mx=("mx", "sum"), my=("my", "sum"))
    c["cx"], c["cy"] = c.mx / c.L, c.my / c.L
    c["centroide_lat"] = c.cy / ky
    c["centroide_lon"] = c.cx / kx
    dx = c.cx.values[:, None] - (depots.lon.values * kx)[None, :]
    dy = c.cy.values[:, None] - (depots.lat.values * ky)[None, :]
    D = np.hypot(dx, dy)
    c["terminal"] = depots.name.values[D.argmin(1)]
    c["dist_centroide_km"] = D.min(1).round(3)
    c = c.reset_index()[[clave, "centroide_lat", "centroide_lon", "terminal", "dist_centroide_km"]]
    t = trips_servicio[["shape_id", "route_id"]].drop_duplicates()
    out = t.merge(c, on=clave)
    return out[["route_id", "shape_id", "centroide_lat", "centroide_lon", "terminal", "dist_centroide_km"]]


def construir_viajes(trips, freqs, stops, st, dist, asign, kwh_km, servicio) -> pd.DataFrame:
    """Expande las ventanas de `frequencies.txt` en viajes individuales."""
    m = freqs.merge(trips[["trip_id", "route_id", "service_id", "shape_id"]], on="trip_id")
    m = m[m.service_id == servicio]
    m = m.merge(asign[["shape_id", "terminal"]], on="shape_id", how="left")
    sin_terminal = m[m.terminal.isna()].shape_id.unique()
    if len(sin_terminal):
        raise ValueError(f"Shapes sin terminal asignado: {sin_terminal[:10]}")

    st = st[st.trip_id.isin(m.trip_id)].copy()
    st["seq"] = st.stop_sequence.astype(int)
    st = st.sort_values(["trip_id", "seq"])
    gb = st.groupby("trip_id")
    info = pd.DataFrame({
        "dur_s": gb.arrival_time.last().map(sec) - gb.departure_time.first().map(sec),
        "stop_origen": gb.stop_id.first(),
        "stop_destino": gb.stop_id.last(),
    }).reset_index()
    xy = stops.set_index("stop_id")
    for k in ["origen", "destino"]:
        info[f"lat_{k}"] = info[f"stop_{k}"].map(xy.stop_lat)
        info[f"lon_{k}"] = info[f"stop_{k}"].map(xy.stop_lon)
    m = m.merge(info, on="trip_id")
    m = m.merge(dist.rename(columns={"distance_km": "km"}), on="shape_id", how="left")

    filas = []
    for r in m.itertuples(index=False):
        for k, dep in enumerate(range(sec(r.start_time), sec(r.end_time), r.headway_secs)):
            filas.append((r.route_id, r.shape_id, r.trip_id, k, r.terminal,
                          dep, dep + r.dur_s, r.km, r.stop_origen, r.stop_destino,
                          r.lat_origen, r.lon_origen, r.lat_destino, r.lon_destino))
    V = pd.DataFrame(filas, columns=[
        "route_id", "shape_id", "trip_id_ventana", "n_salida", "terminal", "dep_s", "arr_s",
        "km", "stop_origen", "stop_destino", "lat_origen", "lon_origen",
        "lat_destino", "lon_destino"])
    V["kwh"] = (V.km * kwh_km).round(2)
    V = V.sort_values(["terminal", "dep_s", "shape_id"]).reset_index(drop=True)
    return V


def partir_bloques(dep_s, objetivo, max_h, gran_min):
    """Cortes horarios [(ini_s, fin_s), ...] que dejan ~`objetivo` viajes por bloque, con
    bloques de a lo más `max_h` horas y cortes en múltiplos de `gran_min` minutos."""
    g = gran_min * 60
    nslots = 24 * 3600 // g
    cnt = np.bincount((dep_s // g).astype(int), minlength=nslots)[:nslots]
    cum = np.concatenate([[0], np.cumsum(cnt)])
    max_k = int(max_h * 3600 // g)
    bloques, i = [], 0
    while i < nslots:
        ks = range(1, min(max_k, nslots - i) + 1)
        k = min(ks, key=lambda k: (abs(cum[i + k] - cum[i] - objetivo), k))
        bloques.append([i * g, (i + k) * g])
        i += k
    if len(bloques) > 1:
        a, b = bloques[-2], bloques[-1]
        n_b = cum[b[1] // g] - cum[b[0] // g]
        if n_b < 0.4 * objetivo and (b[1] - a[0]) <= max_h * 3600:
            bloques[-2] = [a[0], b[1]]
            bloques.pop()
    return [tuple(b) for b in bloques]


def asignar_bloques(V, p):
    """Agrega a `V` las columnas de bloque y devuelve (V, plan) con el plan de bloques."""
    V = V.copy()
    V["bloque"] = ""
    plan = []
    for term, grp in V.groupby("terminal"):
        bl = partir_bloques(grp.dep_s.values, p["objetivo_viajes"], p["max_bloque_h"],
                            p["granularidad_min"])
        for j, (ini, fin) in enumerate(bl, 1):
            sigla = "".join(w[0] for w in term.split()).upper()
            bid = f"{sigla}-{j:02d}"
            idx = grp.index[(grp.dep_s >= ini) & (grp.dep_s < fin)]
            V.loc[idx, "bloque"] = bid
            V.loc[idx, "bloque_ini_s"] = ini
            V.loc[idx, "bloque_fin_s"] = fin
            plan.append(dict(terminal=term, bloque=bid, inicio=hhmm(ini), fin=hhmm(fin),
                             duracion_h=(fin - ini) / 3600, ini_s=ini, fin_s=fin))
    plan = pd.DataFrame(plan)

    V["termina_despues_del_bloque"] = V.arr_s > V.bloque_fin_s
    V["min_fuera_del_bloque"] = ((V.arr_s - V.bloque_fin_s).clip(lower=0) / 60).round(1)

    r = V.groupby("bloque").agg(
        n_viajes=("shape_id", "size"), km_total=("km", "sum"), kwh_total=("kwh", "sum"),
        pct_termina_fuera=("termina_despues_del_bloque", "mean"),
        n_lineas=("route_id", "nunique")).reset_index()
    plan = plan.merge(r, on="bloque")
    plan["pct_termina_fuera"] = (plan.pct_termina_fuera * 100).round(1)
    plan[["km_total", "kwh_total"]] = plan[["km_total", "kwh_total"]].round(1)
    return V, plan


def saturacion_terminales(V, depots) -> pd.DataFrame:
    sat = V.groupby("terminal").size().rename("viajes_dia").reset_index()
    sat = sat.merge(depots[["name", "capacity"]], left_on="terminal", right_on="name").drop(columns="name")
    sat["viajes_por_capacidad"] = (sat.viajes_dia / sat.capacity).round(1)
    return sat.sort_values("viajes_por_capacidad").reset_index(drop=True)


def tabla_viajes(V) -> pd.DataFrame:
    """Viajes en formato de salida, con `viaje_id` = <bloque>-<n correlativo en el bloque>."""
    out = V.copy()
    out["hora_salida"] = out.dep_s.map(hhmm)
    out["hora_llegada"] = out.arr_s.map(hhmm)
    out["dur_min"] = ((out.arr_s - out.dep_s) / 60).round(1)
    out["bloque_ini"] = out.bloque_ini_s.map(hhmm)
    out["bloque_fin"] = out.bloque_fin_s.map(hhmm)
    cols = ["terminal", "bloque", "bloque_ini", "bloque_fin", "route_id", "shape_id",
            "trip_id_ventana", "hora_salida", "hora_llegada", "dep_s", "arr_s", "dur_min",
            "km", "kwh", "stop_origen", "lat_origen", "lon_origen", "stop_destino",
            "lat_destino", "lon_destino", "termina_despues_del_bloque", "min_fuera_del_bloque"]
    out = out[cols].round({"km": 3})
    out.insert(0, "viaje_id", out.bloque + "-" + out.groupby("bloque").cumcount().map(lambda i: f"{i:04d}"))
    return out


def generar(cfg: dict, out_dir: str | Path | None = None, piloto: str | None = None) -> Path:
    """Corre la partición completa y escribe los resultados. Devuelve el Excel piloto."""
    p = cfg["particion"]
    out_dir = Path(out_dir or Path("data/processed") / p["nombre"])
    out_dir.mkdir(parents=True, exist_ok=True)

    gtfs, trips, freqs, stops, st, dist, depots, veh = cargar(cfg)
    kwh_km = veh.battery_kwh / veh.range_km
    cap_util = veh.battery_kwh * (veh.max_soc - veh.min_soc)

    tl = freqs[["trip_id"]].merge(trips[["trip_id", "route_id", "shape_id", "service_id"]], on="trip_id")
    tl = tl[tl.service_id == p["servicio"]]
    asign = asignar_centroide(gtfs, tl, depots, p["nivel"])
    asign.to_csv(out_dir / "asignacion_terminales_centroide.csv", index=False)

    V = construir_viajes(trips, freqs, stops, st, dist, asign, kwh_km, p["servicio"])
    V, plan = asignar_bloques(V, p)
    sat = saturacion_terminales(V, depots)
    out = tabla_viajes(V)

    out.to_csv(out_dir / "viajes_dia_habil.csv", index=False)
    plan.drop(columns=["ini_s", "fin_s"]).to_csv(out_dir / "plan_bloques.csv", index=False)
    sat.to_csv(out_dir / "saturacion_terminales.csv", index=False)

    term = piloto or sat.terminal.iloc[0]
    d0 = sec(p["piloto_desde"] + ":00")
    cand = plan[(plan.terminal == term) & (plan.ini_s >= d0)].iloc[0]
    sub = out[out.bloque == cand.bloque]
    dep = depots[depots.name == term].iloc[0]

    params = pd.DataFrame([
        ["Electroterminal", term, ""],
        ["Criterio de asignación", f"terminal más cercano al centroide del recorrido (nivel: {p['nivel']})", ""],
        ["Bloque", cand.bloque, ""],
        ["Ventana (por hora de salida)", f"{cand.inicio} - {cand.fin}", "hh:mm"],
        ["Duración del bloque", cand.duracion_h, "h"],
        ["Viajes a programar", int(cand.n_viajes), "viajes"],
        ["Capacidad del depósito", int(dep.capacity), "plazas (depots.csv)"],
        ["Lat / Lon depósito", f"{dep.lat}, {dep.lon}", ""],
        ["Batería", veh.battery_kwh, "kWh"],
        ["Autonomía", veh.range_km, "km"],
        ["Consumo específico", round(kwh_km, 3), "kWh/km (batería/autonomía)"],
        ["SoC mínimo / máximo", f"{veh.min_soc} / {veh.max_soc}", ""],
        ["Energía útil", cap_util, "kWh"],
        ["Potencia de carga", veh.charge_power_kw, "kW"],
        ["Buses", "VARIABLE_A_OPTIMIZAR", ""],
        ["Estado inicial de la flota", "buses frescos en el depósito (en bloques posteriores: heredar del bloque anterior)", ""],
    ], columns=["Parámetro", "Valor", "Unidad"])

    por_hora = sub.assign(h=(sub.dep_s // 3600)).groupby("h").agg(
        viajes=("km", "size"), km=("km", "sum"), kwh=("kwh", "sum")).round(1).reset_index()
    por_hora["h"] = por_hora.h.map(lambda h: f"{h:02d}:00")
    por_hora = por_hora.rename(columns={"h": "hora_salida"})

    fname = out_dir / f"subproblema_{cand.bloque}_{term.replace(' ', '_')}.xlsx"
    with pd.ExcelWriter(fname, engine="openpyxl") as w:
        params.to_excel(w, sheet_name="Parametros_Modelo", index=False)
        por_hora.to_excel(w, sheet_name="Resumen_por_hora", index=False)
        sub.drop(columns=["dep_s", "arr_s"]).to_excel(w, sheet_name="Viajes_a_Programar", index=False)
        plan[plan.terminal == term].drop(columns=["ini_s", "fin_s"]).to_excel(
            w, sheet_name="Plan_bloques_terminal", index=False)
    return fname
