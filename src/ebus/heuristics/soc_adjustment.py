"""Sub-rutina de ajuste de SOC integrada en repair (Sección 3.2.1, Tablas 3 y 4 del paper).

Insertar un viaje `c` entre dos viajes `a` y `b` de una cadena exige que el bus llegue a tiempo y que la batería no baje
de `soc_min` en ningún punto: en `c`, en los viajes siguientes, al llegar a cargar y al volver al depósito. Si no alcanza,
se busca dónde cargar (`recharge_opportunity`). `adjust_for_insertion` es el punto de entrada que usa el repair.

Decisiones del equipo (ver docs/decisiones/alns.md):
- **Fórmulas y simulación.** Las fórmulas del paper (`clasificar_insercion`) clasifican el caso (I o II) y el escenario
  (S1, S2, S3); la simulación (`recompute_soc` y `evaluar_cadena`, con nuestra dinámica exacta) confirma siempre. Ninguna
  cadena sale de aquí sin haber recalculado su batería completa.
- **Cargadores y potencia del depósito NO se revisan aquí**, como en el paper: se relajan en el repair y los corrige después
  el ajuste de cargadores y potencia (Paso 7). Solo se respeta la potencia máxima de carga del bus y la del depósito.
- **"Cargar en el intervalo actual"**: primero antes de `c`; si no se puede, después de `c`.
- **Se carga solo lo necesario**, a la máxima potencia disponible, en minutos enteros redondeados hacia arriba y desde que
  el bus llega al depósito.
- `recharge_opportunity` devuelve la cadena modificada (puede extender una carga existente o agregar una nueva).
- No se carga antes del primer viaje (restricción (13)); la energía del retorno al depósito cuenta.

Notación del paper: `a2` = SOC tras `a`; `b1` = SOC antes de `b`; `E_c` = consumo de `c`; `E_lowest` = SOC mínimo desde `b`
(incluido el regreso al depósito). Si `c` va primero, `a` es el depósito de salida; si va último, `b` es el de retorno."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import replace

from ebus.data.fleet import deadhead_energy_soc, trip_energy_soc
from ebus.model.entities import Instance, RechargeEvent, Trip, TripChain, all_recharge_events, recharge_events

_EPS = 1e-9
_MAX_AJUSTES = 8

contador_escenarios: Counter = Counter()


def _e_deadhead(inst: Instance, i: str, j: str) -> float:
    return deadhead_energy_soc(inst.deadhead_distance_km(i, j), inst.bus_type)


def energy_needed(origen: str, c: Trip, destino: str, inst: Instance) -> float:
    """`E_need = E_c + E(d_ac) + E(d_cb)` en fracción de batería: el viaje `c` más los deadheads desde `origen` (donde
    termina `a`, o el depósito de salida si `c` va primero) y hasta `destino` (donde empieza `b`, o el depósito de retorno
    si `c` va último)."""
    return trip_energy_soc(c, inst.bus_type) + _e_deadhead(inst, origen, c.start_station) + _e_deadhead(
        inst, c.end_station, destino)


def recompute_soc(chain: TripChain, inst: Instance) -> None:
    """Recalcula `soc_before` y `soc_after` de toda la cadena, en el lugar (tras destroy o cambios de carga).

    Parte de `inst.soc_initial` en el depósito de salida y recorre los viajes en orden: si hay una carga antes del
    viaje, descuenta la ida al depósito de carga y suma lo cargado (`potencia × duración / batería`, con tope en
    `soc_max`; si hay varios segmentos en el intervalo, los aplica en orden); luego descuenta el deadhead hasta el inicio
    del viaje (`soc_before`) y el viaje (`soc_after`).

    Solo calcula: **no corrige** un SOC que baje de `soc_min` (eso lo detecta el ajuste de SOC o el validador) y no
    mira los horarios de las cargas. El SOC al llegar al depósito al final del día no entra aquí: lo da
    `model/cost.py::soc_at_return`. Una cadena sin viajes queda con las listas vacías."""
    bus = inst.bus_type
    soc = inst.soc_initial
    ubicacion = chain.depot_id
    antes: list[float] = []
    despues: list[float] = []
    for k, trip in enumerate(chain.trips):
        for carga in recharge_events(chain, k):
            soc -= deadhead_energy_soc(inst.deadhead_distance_km(ubicacion, carga.depot_id), bus)
            soc = min(inst.soc_max, soc + carga.power_kw * carga.duration / 60 / bus.battery_kwh)
            ubicacion = carga.depot_id
        soc -= deadhead_energy_soc(inst.deadhead_distance_km(ubicacion, trip.start_station), bus)
        antes.append(soc)
        soc -= trip_energy_soc(trip, bus)
        despues.append(soc)
        ubicacion = trip.end_station
    chain.soc_before[:] = antes
    chain.soc_after[:] = despues


def _retorno(chain: TripChain) -> str:
    return chain.return_depot_id or chain.depot_id


def evaluar_cadena(chain: TripChain, inst: Instance) -> tuple[float, bool]:
    """Recorre la cadena completa y devuelve `(SOC mínimo, ¿horarios factibles?)`.

    El SOC mínimo considera cada punto donde la batería puede tocar fondo: al llegar a cada carga, antes y después de cada
    viaje, y al volver al depósito. Los horarios exigen llegar a tiempo a cada carga y a cada viaje, volver antes del fin del
    horizonte, y no cargar antes del primer viaje (restricción (13)). **No** revisa los límites de cargadores y potencia del
    depósito compartidos con otros buses (ver el docstring del módulo)."""
    bus = inst.bus_type
    soc, lugar, t = inst.soc_initial, chain.depot_id, inst.horizon_start
    minimo, ok = soc, True
    n = len(chain.trips)
    for k in range(n + 1):
        for carga in recharge_events(chain, k):
            if k == 0 and carga.power_kw > 0 and carga.duration > 0:
                ok = False
            if carga.start_time < t + inst.deadhead(lugar, carga.depot_id) or carga.end_time < carga.start_time:
                ok = False
            soc -= _e_deadhead(inst, lugar, carga.depot_id)
            minimo = min(minimo, soc)
            soc = min(inst.soc_max, soc + carga.power_kw * carga.duration / 60 / bus.battery_kwh)
            lugar, t = carga.depot_id, carga.end_time
        if k == n:
            break
        viaje = chain.trips[k]
        if viaje.start_time < t + inst.deadhead(lugar, viaje.start_station):
            ok = False
        soc -= _e_deadhead(inst, lugar, viaje.start_station)
        minimo = min(minimo, soc)
        soc -= trip_energy_soc(viaje, bus)
        minimo = min(minimo, soc)
        lugar, t = viaje.end_station, viaje.end_time
    retorno = _retorno(chain)
    if t + inst.deadhead(lugar, retorno) > inst.horizon_end:
        ok = False
    minimo = min(minimo, soc - _e_deadhead(inst, lugar, retorno))
    return minimo, ok


def es_factible(chain: TripChain, inst: Instance) -> bool:
    """La cadena llega a tiempo a todo y su batería nunca baja de `soc_min` (ver `evaluar_cadena`)."""
    minimo, ok = evaluar_cadena(chain, inst)
    return ok and minimo >= inst.soc_min - _EPS


def _soc_al_volver(chain: TripChain, inst: Instance) -> float:
    """SOC al llegar al depósito de retorno (con la carga final, si la hay). Supone el SOC de la cadena al día."""
    bus = inst.bus_type
    if not chain.trips:
        return inst.soc_initial - _e_deadhead(inst, chain.depot_id, _retorno(chain))
    soc, lugar = chain.soc_after[-1], chain.trips[-1].end_station
    for carga in recharge_events(chain, len(chain.trips)):
        soc -= _e_deadhead(inst, lugar, carga.depot_id)
        soc = min(inst.soc_max, soc + carga.power_kw * carga.duration / 60 / bus.battery_kwh)
        lugar = carga.depot_id
    return soc - _e_deadhead(inst, lugar, _retorno(chain))


def _tiene_carga(chain: TripChain, k: int) -> bool:
    """El intervalo `k` tiene al menos un segmento que carga (un marcador sin potencia ni duración no cuenta)."""
    return any(e.power_kw > 0 and e.duration > 0 for e in recharge_events(chain, k))


def _insertar(chain: TripChain, pos: int, c: Trip, inst: Instance, quitar_cargas: bool = False) -> TripChain:
    """Copia de la cadena con `c` en la posición `pos` y el SOC recalculado.

    Las cargas del intervalo donde entra `c` quedan antes o después de `c` según su horario; las que chocan con `c` se
    descartan. Con `quitar_cargas=True` se descartan todas las de ese intervalo (Caso II, S1)."""
    nueva = chain.copy()
    recargas = {}
    for k, eventos in nueva.recharges.items():
        if k < pos:
            recargas[k] = eventos
        elif k > pos:
            recargas[k + 1] = eventos
    if not quitar_cargas:
        antes, despues = [], []
        for e in recharge_events(nueva, pos):
            if e.end_time + inst.deadhead(e.depot_id, c.start_station) <= c.start_time:
                antes.append(e)
            elif c.end_time + inst.deadhead(c.end_station, e.depot_id) <= e.start_time:
                despues.append(e)
        if antes:
            recargas[pos] = antes
        if despues:
            recargas[pos + 1] = despues
    nueva.trips = chain.trips[:pos] + [c] + chain.trips[pos:]
    nueva.recharges = recargas
    recompute_soc(nueva, inst)
    return nueva


def clasificar_insercion(chain: TripChain, pos: int, c: Trip, inst: Instance) -> tuple[str, str, float]:
    """Caso, escenario y déficit (en fracción de batería) según las fórmulas de las Tablas 3 y 4.

    - Caso I (sin carga entre `a` y `b`): `E_add = E_need + (b1 − a2)` y `E_available = E_lowest − soc_min`.
      S1 si `E_add <= 0`; S2 si `E_available >= E_add`; S3 si no, con déficit `E_add − E_available`.
    - Caso II (con carga entre `a` y `b`): `E_available = E_lowest − soc_min − (b1 − a2)`. S1 si `E_available >= E_need`
      (la carga ya no hace falta); S2 si no, con déficit `E_need − E_available`.

    `b1 − a2` descuenta el deadhead directo `a → b` que la inserción reemplaza (en el Caso I, `a2 − b1` es justamente esa
    energía). El caso se decide por la presencia de cargas en el intervalo, que equivale a comparar `a2` con `b1`."""
    base = chain.copy()
    recompute_soc(base, inst)
    n = len(base.trips)
    origen = base.trips[pos - 1].end_station if pos > 0 else base.depot_id
    destino = base.trips[pos].start_station if pos < n else _retorno(base)
    a2 = base.soc_after[pos - 1] if pos > 0 else inst.soc_initial
    soc_vuelta = _soc_al_volver(base, inst)
    if pos < n:
        b1 = base.soc_before[pos]
        e_lowest = min(base.soc_before[pos:] + base.soc_after[pos:] + [soc_vuelta])
    else:
        b1 = e_lowest = soc_vuelta
    e_need = energy_needed(origen, c, destino, inst)

    if not _tiene_carga(base, pos):
        e_add = e_need + (b1 - a2)
        e_available = e_lowest - inst.soc_min
        if e_add <= _EPS:
            return "I", "S1", 0.0
        if e_available >= e_add - _EPS:
            return "I", "S2", 0.0
        return "I", "S3", e_add - e_available
    e_available = e_lowest - inst.soc_min - (b1 - a2)
    if e_available >= e_need - _EPS:
        return "II", "S1", 0.0
    return "II", "S2", e_need - e_available


def _potencia(inst: Instance, depot_id: str) -> float:
    """Máxima potencia de carga para un bus: la del bus, sin pasar de la del depósito (no se miran otros buses)."""
    deposito = next(d for d in inst.depots if d.id == depot_id)
    return min(inst.bus_type.max_charge_kw, deposito.power_kw)


def _siguiente(chain: TripChain, k: int, inst: Instance) -> tuple[str, int]:
    """Dónde y cuándo debe estar el bus al terminar el intervalo `k`: el inicio del viaje `k`, o el depósito de retorno
    al fin del horizonte si `k` es el intervalo final."""
    if k < len(chain.trips):
        return chain.trips[k].start_station, chain.trips[k].start_time
    return _retorno(chain), inst.horizon_end


def _con_carga_nueva(chain: TripChain, k: int, minutos: int, inst: Instance) -> TripChain | None:
    """Agrega una visita nueva al depósito más cercano en el intervalo `k` (entre los viajes `k−1` y `k`), con `minutos`
    de carga desde que el bus llega. `None` si no cabe en la holgura."""
    if minutos <= 0:
        return None
    previo = chain.trips[k - 1]
    deposito = inst.nearest_depot(previo.end_station)
    lugar, t_siguiente = _siguiente(chain, k, inst)
    llegada = previo.end_time + inst.deadhead(previo.end_station, deposito.id)
    if llegada + minutos + inst.deadhead(deposito.id, lugar) > t_siguiente:
        return None
    nueva = chain.copy()
    nueva.recharges[k] = [RechargeEvent(deposito.id, llegada, llegada + minutos, _potencia(inst, deposito.id))]
    recompute_soc(nueva, inst)
    return nueva


def _con_carga_extendida(chain: TripChain, k: int, minutos: int, inst: Instance) -> TripChain | None:
    """Extiende `minutos` el último segmento del intervalo `k`. Si es un marcador de visita sin carga, pasa a cargar desde
    su inicio. `None` si ya no alcanza a llegar a tiempo al viaje siguiente (o a volver al depósito)."""
    if minutos <= 0:
        return None
    nueva = chain.copy()
    eventos = list(recharge_events(nueva, k))
    ultimo = eventos[-1]
    if ultimo.power_kw > 0 and ultimo.duration > 0:
        extendido = replace(ultimo, end_time=ultimo.end_time + minutos)
    else:
        extendido = replace(ultimo, power_kw=_potencia(inst, ultimo.depot_id),
                            end_time=max(ultimo.end_time, ultimo.start_time + minutos))
    lugar, t_siguiente = _siguiente(nueva, k, inst)
    if extendido.end_time + inst.deadhead(extendido.depot_id, lugar) > t_siguiente:
        return None
    eventos[-1] = extendido
    nueva.recharges[k] = eventos
    recompute_soc(nueva, inst)
    return nueva


def _ajustar(construir: Callable[[int], TripChain | None], potencia: float, deficit: float,
             inst: Instance) -> TripChain | None:
    """Busca los minutos de carga justos: parte de los minutos que cubren `deficit` a `potencia` y, si la simulación
    todavía muestra un SOC bajo el mínimo, agrega los minutos que faltan. Se rinde si la carga ya no cabe en la holgura,
    si los horarios dejan de ser factibles o tras `_MAX_AJUSTES` vueltas (p. ej., si el déficit ocurre antes de la carga)."""
    bateria = inst.bus_type.battery_kwh
    minutos = max(1, math.ceil(deficit * bateria / potencia * 60 - _EPS))
    for _ in range(_MAX_AJUSTES):
        candidata = construir(minutos)
        if candidata is None:
            return None
        minimo, horarios_ok = evaluar_cadena(candidata, inst)
        if not horarios_ok:
            return None
        if minimo >= inst.soc_min - _EPS:
            return candidata
        minutos += max(1, math.ceil((inst.soc_min - minimo) * bateria / potencia * 60 - _EPS))
    return None


def _intentar_extender(chain: TripChain, k: int, deficit: float, inst: Instance) -> TripChain | None:
    """Carga más extendiendo la visita al depósito que ya existe en el intervalo `k`."""
    if not 1 <= k <= len(chain.trips) or not recharge_events(chain, k):
        return None
    ultimo = recharge_events(chain, k)[-1]
    potencia = ultimo.power_kw if ultimo.power_kw > 0 and ultimo.duration > 0 else _potencia(inst, ultimo.depot_id)
    return _ajustar(lambda m: _con_carga_extendida(chain, k, m, inst), potencia, deficit, inst)


def _intentar_nueva(chain: TripChain, k: int, deficit: float, inst: Instance) -> TripChain | None:
    """Carga en una visita nueva al depósito en el intervalo `k`, que no debe tener visitas. El déficit se amplía con la
    energía extra del desvío al depósito."""
    if not 1 <= k <= len(chain.trips) or recharge_events(chain, k):
        return None
    previo = chain.trips[k - 1]
    deposito = inst.nearest_depot(previo.end_station)
    lugar, _ = _siguiente(chain, k, inst)
    desvio = (_e_deadhead(inst, previo.end_station, deposito.id) + _e_deadhead(inst, deposito.id, lugar)
              - _e_deadhead(inst, previo.end_station, lugar))
    return _ajustar(lambda m: _con_carga_nueva(chain, k, m, inst), _potencia(inst, deposito.id),
                    deficit + desvio, inst)


def recharge_opportunity(chain: TripChain, pos: int, deficit: float, inst: Instance) -> TripChain | None:
    """Agrega carga a una cadena que ya contiene a `c` en la posición `pos` y le falta `deficit` de batería.

    Prueba, en el orden del paper, y devuelve la primera cadena factible (o `None`):
    1. extender una carga del intervalo anterior a `c`;
    2. cargar en el intervalo actual, en el depósito más cercano: primero antes de `c`, luego después de `c`
       (extendiendo una visita que ya esté ahí, o con una nueva);
    3. extender visitas al depósito de intervalos anteriores, del más cercano a `c` al más lejano;
    4. crear una visita nueva en intervalos anteriores sin visitas y, si no, en los posteriores.
    Nunca carga antes del primer viaje (restricción (13)). La cantidad la afina `_ajustar` simulando."""
    n = len(chain.trips)
    deficit = max(deficit, 0.0)
    intentos = [lambda: _intentar_extender(chain, pos, deficit, inst),
                lambda: _intentar_nueva(chain, pos, deficit, inst),
                lambda: _intentar_extender(chain, pos + 1, deficit, inst),
                lambda: _intentar_nueva(chain, pos + 1, deficit, inst)]
    intentos += [lambda k=k: _intentar_extender(chain, k, deficit, inst) for k in range(pos - 1, 0, -1)]
    intentos += [lambda k=k: _intentar_nueva(chain, k, deficit, inst) for k in range(pos - 1, 0, -1)]
    intentos += [lambda k=k: _intentar_nueva(chain, k, deficit, inst) for k in range(pos + 2, n + 1)]
    for intento in intentos:
        resultado = intento()
        if resultado is not None:
            return resultado
    return None


def adjust_case_one(chain: TripChain, pos: int, c: Trip, inst: Instance) -> TripChain | None:
    """Caso I (sin carga entre `a` y `b`), Tabla 3. S1 y S2: `c` entra sin cargar. S3: hace falta cargar
    (`recharge_opportunity`). La simulación decide: si las fórmulas dicen S3 pero la batería alcanza (p. ej., porque una
    carga posterior llega igual al tope), se inserta sin cargar."""
    candidata = _insertar(chain, pos, c, inst)
    minimo, horarios_ok = evaluar_cadena(candidata, inst)
    if not horarios_ok:
        return None
    if minimo >= inst.soc_min - _EPS:
        return candidata
    return recharge_opportunity(candidata, pos, inst.soc_min - minimo, inst)


def adjust_case_two(chain: TripChain, pos: int, c: Trip, inst: Instance) -> TripChain | None:
    """Caso II (había carga entre `a` y `b`), Tabla 4. S1: si sin esa carga igual alcanza, se elimina. S2: si no, se
    conservan los segmentos que siguen cabiendo antes o después de `c` y se carga más, priorizando extender esos segmentos;
    si no basta, `recharge_opportunity` desde el paso 1."""
    sin_carga = _insertar(chain, pos, c, inst, quitar_cargas=True)
    if es_factible(sin_carga, inst):
        return sin_carga
    con_carga = _insertar(chain, pos, c, inst)
    minimo, horarios_ok = evaluar_cadena(con_carga, inst)
    if not horarios_ok:
        return None
    if minimo >= inst.soc_min - _EPS:
        return con_carga
    deficit = inst.soc_min - minimo
    for k in (pos, pos + 1):
        resultado = _intentar_extender(con_carga, k, deficit, inst)
        if resultado is not None:
            return resultado
    return recharge_opportunity(con_carga, pos, deficit, inst)


def adjust_for_insertion(chain: TripChain, pos: int, c: Trip, inst: Instance) -> TripChain | None:
    """Cadena nueva con `c` insertado en la posición `pos` (entre `trips[pos−1]` y `trips[pos]`), con su SOC y sus cargas
    ajustados, o `None` si no se puede. No modifica `chain`.

    Primero descarta rápido lo imposible por tiempo (si `c` no cabe entre `a` y `b` ni siquiera sin cargar, ninguna carga lo
    arregla). Después las fórmulas del paper (`clasificar_insercion`) eligen el caso (I o II) y predicen el escenario; la
    simulación decide el resultado, y `contador_escenarios` registra si confirmó o corrigió la predicción."""
    n = len(chain.trips)
    if not 0 <= pos <= n:
        raise ValueError(f"posición {pos} fuera de la cadena de {n} viajes")
    if any(t.id == c.id for t in chain.trips):
        raise ValueError(f"el viaje {c.id} ya está en la cadena")
    if pos > 0:
        lugar, t = chain.trips[pos - 1].end_station, chain.trips[pos - 1].end_time
    else:
        lugar, t = chain.depot_id, inst.horizon_start
    if c.start_time < t + inst.deadhead(lugar, c.start_station):
        return None
    destino, t_destino = _siguiente(chain, pos, inst)
    if c.end_time + inst.deadhead(c.end_station, destino) > t_destino:
        return None
    caso, escenario, _ = clasificar_insercion(chain, pos, c, inst)
    resultado = adjust_case_two(chain, pos, c, inst) if caso == "II" else adjust_case_one(chain, pos, c, inst)
    if resultado is None:
        contador_escenarios[(caso, escenario, "imposible")] += 1
    else:
        cargado = sum(e.power_kw * e.duration for e in all_recharge_events(resultado))
        cargado_antes = sum(e.power_kw * e.duration for e in all_recharge_events(chain))
        contador_escenarios[(caso, escenario, "con carga" if cargado > cargado_antes + _EPS else "sin carga")] += 1
    return resultado
