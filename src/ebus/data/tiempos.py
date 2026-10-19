from __future__ import annotations


def salida_a_minutos(segundos: int) -> int:
    return int(segundos) // 60


def llegada_a_minutos(segundos: int) -> int:
    return -(-int(segundos) // 60)


def hhmm_a_minutos(texto: str) -> int:
    horas, minutos = str(texto).split(":")[:2]
    return int(horas) * 60 + int(minutos)
