from __future__ import annotations

from ebus.model.entities import Instance, Solution
from ebus.model.network import TimeSpaceNetwork


def plot_schedule(sol: Solution, inst: Instance, path: str | None = None) -> None:
    raise NotImplementedError


def plot_soc_profiles(sol: Solution, inst: Instance, path: str | None = None) -> None:
    raise NotImplementedError


def plot_network(net: TimeSpaceNetwork, path: str | None = None) -> None:
    raise NotImplementedError


def plot_convergence(history: list[dict], path: str | None = None) -> None:
    raise NotImplementedError
