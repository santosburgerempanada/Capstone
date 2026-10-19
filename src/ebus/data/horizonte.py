from __future__ import annotations

from ebus.model.entities import Trip

REGLA_MAX_DEADHEAD = "max_deadhead"


def margen_horizonte(deadhead_min: dict[tuple[str, str], int], regla: str | int) -> int:
    if regla == REGLA_MAX_DEADHEAD:
        if not deadhead_min:
            raise ValueError("La matriz de deadheads está vacía: no se puede calcular el margen")
        return max(deadhead_min.values())
    if isinstance(regla, int) and not isinstance(regla, bool) and regla >= 0:
        return regla
    raise ValueError(f"network.horizon_margin desconocido: {regla!r}")


def calcular_horizonte(viajes: list[Trip], inicio_bloque: int, margen: int) -> tuple[int, int]:
    if not viajes:
        raise ValueError("El subproblema no tiene viajes")
    fuera = [v.id for v in viajes if v.start_time < inicio_bloque]
    if fuera:
        raise ValueError(f"Viajes que salen antes del inicio del bloque: {fuera[:5]}")
    ultima_llegada = max(v.end_time for v in viajes)
    return inicio_bloque - margen, ultima_llegada + margen
