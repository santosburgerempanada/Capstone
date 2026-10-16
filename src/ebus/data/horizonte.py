"""Horizonte T0–T-1 de un subproblema de la partición (ver docs/decisiones/instancias.md, "Viajes que
terminan fuera del bloque").

Los viajes de un bloque son los que salen dentro de él, pero muchos terminan después de que el
bloque cierra. Por eso el horizonte se extiende en ambos extremos: antes del bloque, para que un
bus alcance a salir del depósito y llegar al primer viaje; después de la última llegada, para que
todos los buses alcancen a volver. El margen se calcula desde la matriz de deadheads, así que si
la matriz cambia, el horizonte cambia con ella."""

from __future__ import annotations

from ebus.model.entities import Trip

REGLA_MAX_DEADHEAD = "max_deadhead"


def margen_horizonte(deadhead_min: dict[tuple[str, str], int], regla: str | int) -> int:
    """Minutos que se extiende el horizonte a cada lado del bloque.

    `regla` viene de `network.horizon_margin` en la config: `"max_deadhead"` usa el deadhead más
    largo de la matriz (conservador: cubre cualquier salida o retorno al depósito); un entero
    fija el margen en minutos."""
    if regla == REGLA_MAX_DEADHEAD:
        if not deadhead_min:
            raise ValueError("La matriz de deadheads está vacía: no se puede calcular el margen")
        return max(deadhead_min.values())
    if isinstance(regla, int) and not isinstance(regla, bool) and regla >= 0:
        return regla
    raise ValueError(f"network.horizon_margin desconocido: {regla!r}")


def calcular_horizonte(viajes: list[Trip], inicio_bloque: int, margen: int) -> tuple[int, int]:
    """(horizon_start, horizon_end) = (inicio del bloque − margen, última llegada + margen).

    No se recorta a [0, 1440): un bloque que parte a las 00:00 empieza en minutos negativos (el
    día anterior) y uno nocturno puede terminar después de 1440. `Tariff.price_at` ya trabaja
    módulo 1440."""
    if not viajes:
        raise ValueError("El subproblema no tiene viajes")
    fuera = [v.id for v in viajes if v.start_time < inicio_bloque]
    if fuera:
        raise ValueError(f"Viajes que salen antes del inicio del bloque: {fuera[:5]}")
    ultima_llegada = max(v.end_time for v in viajes)
    return inicio_bloque - margen, ultima_llegada + margen
