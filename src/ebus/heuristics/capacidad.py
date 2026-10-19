from __future__ import annotations

import numpy as np

from ebus.model.entities import Instance, TripChain, all_recharge_events

_EPS = 1e-6


class CapacidadLibre:

    def __init__(self, inst: Instance):
        self.inicio = inst.horizon_start
        largo = max(inst.horizon_end - inst.horizon_start, 0)
        self._cargadores = {d.id: np.full(largo, d.n_chargers, dtype=np.int64) for d in inst.depots}
        self._potencia = {d.id: np.full(largo, float(d.power_kw)) for d in inst.depots}

    def _tramo(self, desde: int, hasta: int) -> slice:
        return slice(desde - self.inicio, hasta - self.inicio)

    def ocupar(self, cadena: TripChain) -> None:
        for carga in all_recharge_events(cadena):
            if carga.end_time <= carga.start_time:
                continue
            tramo = self._tramo(carga.start_time, carga.end_time)
            self._cargadores[carga.depot_id][tramo] -= 1
            self._potencia[carga.depot_id][tramo] -= carga.power_kw
            if self._cargadores[carga.depot_id][tramo].min() < 0 or self._potencia[carga.depot_id][tramo].min() < -_EPS:
                raise RuntimeError(f"La cadena {cadena.bus_id} excede la capacidad libre del depósito {carga.depot_id} "
                                   f"entre {carga.start_time} y {carga.end_time}")

    def cargadores_libres(self, depot_id: str, desde: int, hasta: int) -> int:
        valores = self._cargadores[depot_id][self._tramo(desde, hasta)]
        return int(valores.min()) if len(valores) else np.iinfo(np.int64).max

    def potencia_libre(self, depot_id: str, desde: int, hasta: int) -> float:
        valores = self._potencia[depot_id][self._tramo(desde, hasta)]
        return float(valores.min()) if len(valores) else float("inf")
