"""Conversión de horas en segundos a minutos enteros (convención del proyecto: minutos desde las 00:00).

Los datos de la partición traen segundos (`dep_s`, `arr_s` de `viajes_dia_habil.csv`), pero la
`Instance` trabaja en minutos enteros. El redondeo es conservador para la conectividad entre
viajes: la salida se redondea hacia abajo y la llegada hacia arriba, así un bus nunca parece llegar
antes ni salir después de lo real. El costo es que un viaje puede figurar hasta 2 min más largo.
Ver docs/decisiones/datos.md."""

from __future__ import annotations


def salida_a_minutos(segundos: int) -> int:
    """Hora de salida en minutos, redondeada hacia abajo."""
    return int(segundos) // 60


def llegada_a_minutos(segundos: int) -> int:
    """Hora de llegada en minutos, redondeada hacia arriba."""
    return -(-int(segundos) // 60)


def hhmm_a_minutos(texto: str) -> int:
    """"14:30" -> 870. Acepta horas de 24 o más ("24:00" -> 1440)."""
    horas, minutos = str(texto).split(":")[:2]
    return int(horas) * 60 + int(minutos)
