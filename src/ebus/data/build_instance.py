from __future__ import annotations

import pickle
from pathlib import Path

from ebus.data.subproblema import construir_instancia_subproblema
from ebus.data.synthetic import generate_instance, paper_figure1_instance
from ebus.model.entities import Instance


def build_instance(cfg: dict) -> Instance:
    data_cfg = cfg.get("data", {})
    battery_cfg = cfg.get("battery", {})
    consumo = {
        "consumption_kwh_per_km": battery_cfg.get("consumption_kwh_per_km"),
        "deadhead_consumption_factor": battery_cfg.get("deadhead_consumption_factor"),
        "soc_initial": battery_cfg.get("soc_initial"),
        "deadhead_cost_per_km": cfg.get("costs", {}).get("deadhead_cost_per_km"),
    }
    source = data_cfg.get("source")

    if source == "synthetic":
        synth_cfg = data_cfg.get("synthetic", {})
        kind = synth_cfg.get("kind", "random")
        if kind == "figure1":
            return paper_figure1_instance(**consumo)
        if kind == "random":
            return generate_instance(
                n_trips=synth_cfg["n_trips"],
                n_stations=synth_cfg["n_stations"],
                n_depots=synth_cfg["n_depots"],
                seed=synth_cfg.get("seed", 0),
                battery_kwh=synth_cfg.get("battery_kwh"),
                n_chargers=synth_cfg.get("n_chargers"),
                **consumo,
            )
        raise ValueError(f"data.synthetic.kind desconocido: {kind!r}")

    if source == "subproblema":
        return construir_instancia_subproblema(cfg)

    raise NotImplementedError(
        "Solo están implementadas las fuentes `synthetic` y `subproblema` (un bloque de la partición); "
        "la construcción desde el GTFS completo no lo está. Ver docs/datos/datos_requeridos.md."
    )


def save_instance(instance: Instance, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(instance, f)


def load_instance(path: str | Path) -> Instance:
    with Path(path).open("rb") as f:
        return pickle.load(f)
