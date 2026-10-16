"""Bucle principal del ALNS (Apéndice A, Tabla A1, y flujo de la Fig. 2 del paper), sobre la librería `alns`.

Cada iteración: la ruleta elige un destroy y un repair (A.6); el destroy quita viajes; el repair los reinserta (con ajuste
de SOC) y la post-reparación cubre los que no cupieron; la candidata se valida, se acepta o no según A.1 y, según el
resultado, se actualizan los pesos de los operadores (A.7, A.8).

Decisiones del equipo (ver docs/decisiones/alns.md):
- **D1:** el ciclo es el de la librería `alns` (`ALNS.iterate`), con piezas propias: el estado (`EstadoAlns`), la ruleta
  con reinicio de pesos cada 100 iteraciones (`RuletaConReinicio`) y el criterio de aceptación de A.1 (`AceptacionA1`).
  El "repair" que ve la librería incluye la post-reparación y el cierre de la solución.
- **D2a:** `Pr_accept` es el **factor de enfriamiento por iteración**: `T ← Pr_accept · T` (el paper no dice cómo baja la
  temperatura; con `T` constante `Pr_accept` no tendría efecto, lo que contradice su Fig. 8).
- **D2b:** la candidata se compara con la **mejor** solución encontrada, como en A.1.
- **D3:** toda candidata se valida con `check_solution`. Si todas sus violaciones son de capacidad de estacionamiento, se
  rechaza (objetivo infinito); cualquier otra violación es un bug y detiene el ALNS.
- **Paso 7:** entre el repair y la post-reparación corre el ajuste de cargadores y potencia (`charger_power`), que deja
  la candidata dentro de los límites de cada depósito; los viajes que quita van a la post-reparación junto con los que el
  repair no insertó. (Reemplaza a la regla temporal D4, que rechazaba las candidatas que excedían los cargadores.)
- **D5:** `N_max` iteraciones (15 000 en la config, como el paper)."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from alns import ALNS
from alns.accept import AcceptanceCriterion
from alns.Outcome import Outcome
from alns.select import RouletteWheel
from alns.stop import MaxIterations

from ebus.evaluation.feasibility import check_solution
from ebus.heuristics.charger_power import enforce_charging_resources
from ebus.heuristics.destroy import operadores_destroy
from ebus.heuristics.post_repair import cerrar_solucion
from ebus.heuristics.repair import operadores_repair
from ebus.model.cost import evaluate
from ebus.model.entities import Instance, Solution, Trip

_RESULTADOS = {Outcome.BEST: "nueva mejor", Outcome.BETTER: "mejora la actual", Outcome.ACCEPT: "aceptada",
               Outcome.REJECT: "rechazada"}


@dataclass
class AlnsParams:
    max_iterations: int = 15000
    destroy_max: int = 5
    accept_probability: float = 0.99
    initial_temperature_factor: float = 0.05
    reaction_factor: float = 0.5
    scores: tuple[float, float, float, float] = (3, 2, 1, 0)
    weight_reset_every: int = 100
    seed: int = 42
    post_repair_time_limit_s: float = 30.0
    reinicio_mejor_cada: int | None = None

    @classmethod
    def desde_config(cls, alns_cfg: dict) -> AlnsParams:
        """Parámetros desde la sección `alns` de la config (los valores por defecto de la clase no se usan)."""
        return cls(
            max_iterations=alns_cfg["max_iterations"], destroy_max=alns_cfg["destroy_max"],
            accept_probability=alns_cfg["accept_probability"],
            initial_temperature_factor=alns_cfg["initial_temperature_factor"],
            reaction_factor=alns_cfg["reaction_factor"], scores=tuple(alns_cfg["scores"]),
            weight_reset_every=alns_cfg["weight_reset_every"], seed=alns_cfg["seed"],
            post_repair_time_limit_s=alns_cfg["post_repair"]["time_limit_s"],
            reinicio_mejor_cada=alns_cfg.get("reinicio_mejor_cada"),
        )


def initial_temperature(initial_cost: float, factor: float = 0.05) -> float:
    """(A.2): T0 = -T_ini · 0.05 / ln 0.5."""
    return -initial_cost * factor / math.log(0.5)


def acceptance_probability(best_cost: float, new_cost: float, temperature: float) -> float:
    """(A.1): Pr = exp((f(x_b) - f(x_t)) / T)."""
    return math.exp((best_cost - new_cost) / temperature)


@dataclass
class EstadoAlns:
    """Estado que maneja la librería. Tras un destroy guarda la solución destruida y los viajes quitados; tras el repair,
    la solución completa y cuántos viajes quitó el ajuste de cargadores. `objective()` es el costo total."""

    solucion: Solution
    inst: Instance
    quitados: list[Trip] = field(default_factory=list)
    quitados_por_cargadores: int = 0
    excede_capacidad: bool = False
    desde_mejor: bool = False

    def objective(self) -> float:
        if self.excede_capacidad:
            return math.inf
        if self.solucion.cost is None:
            self.solucion.cost = evaluate(self.solucion, self.inst).total
        return self.solucion.cost


def _destroy_alns(operador: Callable, d_max: int, memoria: dict | None = None) -> Callable:
    """Adapta un destroy a la librería. Con `memoria` (`{"iteracion", "mejor", "reinicio_cada"}`, compartida por todos
    los destroy), cada `reinicio_cada` iteraciones destruye la mejor solución encontrada en vez de la actual (extensión,
    no es del paper): una búsqueda que se alejó de la mejor vuelve a partir de ella (H23)."""
    def destroy(estado: EstadoAlns, rng: np.random.Generator, **_) -> EstadoAlns:
        base, desde_mejor = estado, False
        if memoria is not None:
            memoria["iteracion"] += 1
            cada = memoria.get("reinicio_cada")
            if cada and memoria["iteracion"] % cada == 0:
                base, desde_mejor = memoria["mejor"], True
        destruida, quitados = operador(base.solucion, base.inst, d_max, rng)
        return EstadoAlns(destruida, base.inst, quitados, desde_mejor=desde_mejor)
    return destroy


def _repair_alns(operador: Callable, time_limit_s: float) -> Callable:
    """Repair + ajuste de cargadores y potencia (Paso 7) + post-reparación + cierre, y la validación de cada candidata
    (D3)."""
    def repair(estado: EstadoAlns, rng: np.random.Generator, **_) -> EstadoAlns:
        reparada, pendientes = operador(estado.solucion, estado.quitados, estado.inst, rng)
        ajustada, quitados = enforce_charging_resources(reparada, estado.inst, rng)
        cerrada = cerrar_solucion(ajustada, pendientes + quitados, estado.inst, time_limit_s)
        violaciones = check_solution(cerrada, estado.inst)
        de_capacidad = [v for v in violaciones if "estacionados" in v]
        if violaciones and len(de_capacidad) == len(violaciones):
            return EstadoAlns(cerrada, estado.inst, quitados_por_cargadores=len(quitados), excede_capacidad=True,
                              desde_mejor=estado.desde_mejor)
        if violaciones:
            raise RuntimeError(
                "El repair entregó una solución inválida; es un bug (CLAUDE.md: se corrige la causa, no el validador): "
                + "; ".join(violaciones[:3])
            )
        return EstadoAlns(cerrada, estado.inst, quitados_por_cargadores=len(quitados), desde_mejor=estado.desde_mejor)
    return repair


class AceptacionA1(AcceptanceCriterion):
    """(A.1) con enfriamiento geométrico por iteración (D2a). Se acepta siempre una candidata igual o mejor que la mejor
    encontrada; si no, con probabilidad `exp((f(x_b) − f(x_t)) / T)` (D2b: contra la mejor). Tras cada llamada (una por
    iteración) `T ← factor · T`."""

    def __init__(self, temperatura_inicial: float, factor_enfriamiento: float):
        if temperatura_inicial <= 0 or not 0 < factor_enfriamiento <= 1:
            raise ValueError("La temperatura debe ser positiva y el factor de enfriamiento estar en (0, 1]")
        self.temperatura = temperatura_inicial
        self.factor = factor_enfriamiento

    def __call__(self, rng: np.random.Generator, best, current, candidate) -> bool:
        temperatura = self.temperatura
        self.temperatura *= self.factor
        costo = candidate.objective()
        if costo <= best.objective():
            return True
        if temperatura <= 0:
            return False
        return bool(rng.random() < acceptance_probability(best.objective(), costo, temperatura))


class RuletaConReinicio(RouletteWheel):
    """Ruleta de la librería (A.6; actualización `ρ ← decay·ρ + (1 − decay)·σ`, que con `decay = 1 − λ` es A.8) más el
    reinicio de todos los pesos a 1 cada `reinicio_cada` iteraciones, como el paper. `al_actualizar` permite registrar
    cada iteración."""

    def __init__(self, scores, decay: float, num_destroy: int, num_repair: int, reinicio_cada: int,
                 al_actualizar: Callable | None = None):
        super().__init__(list(scores), decay, num_destroy, num_repair)
        self.reinicio_cada = reinicio_cada
        self.iteracion = 0
        self.al_actualizar = al_actualizar

    def update(self, cand, d_idx: int, r_idx: int, outcome) -> None:
        super().update(cand, d_idx, r_idx, outcome)
        self.iteracion += 1
        if self.al_actualizar is not None:
            self.al_actualizar(cand, d_idx, r_idx, outcome)
        if self.iteracion % self.reinicio_cada == 0:
            self._d_weights[:] = 1.0
            self._r_weights[:] = 1.0


@dataclass
class ResultadoAlns:
    mejor: Solution
    historia: list[dict]
    tiempo_s: float
    costo_inicial: float


def run_alns(initial: Solution, inst: Instance, alns_cfg: dict, iteraciones: int | None = None,
             semilla: int | None = None) -> ResultadoAlns:
    """Corre el ALNS desde `initial` (hoy, el baseline: decisión provisoria del equipo) con los parámetros de la sección
    `alns` de la config; `iteraciones` y `semilla` reemplazan a `max_iterations` y `seed` si se dan. Devuelve la mejor
    solución (completa y validada) y la historia de cada iteración."""
    params = AlnsParams.desde_config(alns_cfg)
    if iteraciones is not None:
        params.max_iterations = iteraciones
    if semilla is not None:
        params.seed = semilla

    inicial = initial.copy()
    inicial.cost = evaluate(inicial, inst).total
    violaciones = check_solution(inicial, inst)
    if violaciones:
        raise ValueError(f"La solución inicial no es válida: {violaciones[:3]}")

    destroys = operadores_destroy(alns_cfg)
    repairs = operadores_repair(alns_cfg)
    motor = ALNS(np.random.default_rng(params.seed))
    memoria = {"iteracion": 0, "mejor": EstadoAlns(inicial, inst), "reinicio_cada": params.reinicio_mejor_cada}
    for nombre, operador in destroys.items():
        motor.add_destroy_operator(_destroy_alns(operador, params.destroy_max, memoria), nombre)
    for nombre, operador in repairs.items():
        motor.add_repair_operator(_repair_alns(operador, params.post_repair_time_limit_s), nombre)

    aceptacion = AceptacionA1(initial_temperature(inicial.cost, params.initial_temperature_factor),
                              params.accept_probability)
    historia: list[dict] = []
    seguimiento = {"actual": inicial.cost, "mejor": inicial.cost, "buses_mejor": inicial.n_buses}
    nombres_d, nombres_r = list(destroys), list(repairs)

    def registrar(cand: EstadoAlns, d_idx: int, r_idx: int, outcome) -> None:
        costo = cand.objective()
        if outcome == Outcome.BEST:
            seguimiento.update(actual=costo, mejor=costo, buses_mejor=cand.solucion.n_buses)
            memoria["mejor"] = cand
        elif outcome in (Outcome.BETTER, Outcome.ACCEPT):
            seguimiento["actual"] = costo
        historia.append({
            "iteracion": len(historia) + 1,
            "destroy": nombres_d[d_idx],
            "repair": nombres_r[r_idx],
            "resultado": _RESULTADOS[outcome],
            "costo_candidata": round(costo, 4),
            "buses_candidata": cand.solucion.n_buses,
            "quitados_por_cargadores": cand.quitados_por_cargadores,
            "excede_capacidad": cand.excede_capacidad,
            "desde_mejor": cand.desde_mejor,
            "costo_actual": round(seguimiento["actual"], 4),
            "costo_mejor": round(seguimiento["mejor"], 4),
            "buses_mejor": seguimiento["buses_mejor"],
            "temperatura": aceptacion.temperatura / params.accept_probability,
        })

    ruleta = RuletaConReinicio(params.scores, 1 - params.reaction_factor, len(destroys), len(repairs),
                               params.weight_reset_every, registrar)
    inicio = time.perf_counter()
    resultado = motor.iterate(EstadoAlns(inicial, inst), ruleta, aceptacion, MaxIterations(params.max_iterations))
    tiempo = time.perf_counter() - inicio

    mejor = resultado.best_state.solucion
    finales = check_solution(mejor, inst)
    if finales:
        raise RuntimeError(f"La mejor solución del ALNS no es válida; es un bug: {finales[:3]}")
    return ResultadoAlns(mejor=mejor, historia=historia, tiempo_s=tiempo, costo_inicial=inicial.cost)
