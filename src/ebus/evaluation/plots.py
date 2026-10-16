"""Gráficos de resultados."""

from __future__ import annotations

from ebus.model.entities import Instance, Solution
from ebus.model.network import TimeSpaceNetwork


def plot_schedule(sol: Solution, inst: Instance, path: str | None = None) -> None:
    """Diagrama tipo Gantt de viajes, vacíos y cargas por bus (Fig. 7)."""
    raise NotImplementedError


def plot_soc_profiles(sol: Solution, inst: Instance, path: str | None = None) -> None:
    """Evolución del SOC de cada cadena (Fig. 3)."""
    raise NotImplementedError


def plot_network(net: TimeSpaceNetwork, path: str | None = None) -> None:
    """Red espacio-tiempo (Fig. 1); útil para depurar instancias pequeñas."""
    raise NotImplementedError


def plot_convergence(history: list[dict], path: str | None = None) -> None:
    """Costo actual / mejor por iteración del ALNS."""
    raise NotImplementedError
