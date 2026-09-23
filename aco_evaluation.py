"""Auswertung der Ant-System-Demo: ein Lauf, Sweep über Ameisenzahl/Verdunstungsrate ρ, und zwei Experimente -
Kopfexperiment (eigenständig gemessene Abweichung vom Brute-Force-Optimum auf einer kleinen Instanz) und eigener
Regler (Verdunstungsrate ρ)."""

from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import permutations

import numpy as np

import aco_algorithm as A
import aco_constants as C
import aco_scenario as S

BRUTE_FORCE_MAX_N = 9      # (n-1)! Touren; bei 9 sind das 40 320 - noch < 1 s


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED
    ants: int = C.DEFAULT_ANTS
    gens: int = C.DEFAULT_GEN
    alpha: float = C.DEFAULT_ALPHA
    beta: float = C.DEFAULT_BETA
    rho: float = C.DEFAULT_RHO
    run_seed: int = C.DEFAULT_RUN_SEED


@lru_cache(maxsize=256)
def instance(n, seed):
    inst = S.generate_perm(n, 0, seed)
    return inst, A.dist_matrix(inst.xy)


def run(settings, keep_history=False):
    inst, D = instance(settings.n, settings.seed)
    return A.run_aco(D, settings.ants, settings.gens, settings.alpha, settings.beta, settings.rho, C.Q, C.TAU0, settings.run_seed, keep_history=keep_history)


@dataclass
class Analysis:
    settings: Settings
    result: object
    inst: object
    D: np.ndarray
    brute_force_optimum: float             # nur bei n <= BRUTE_FORCE_MAX_N berechnet, sonst NaN

    @property
    def gap(self):
        if not np.isfinite(self.brute_force_optimum) or self.brute_force_optimum == 0:
            return float("nan")
        # ACO kann das echte Optimum nie unterbieten - ein winzig negativer Wert ist reines Fließkomma-Rauschen
        # (z. B. bei bitidentischer Tour), auf 0 geklemmt statt als verwirrende Mikro-Zahl angezeigt
        return max(0.0, 100.0 * (self.result.best_length - self.brute_force_optimum) / self.brute_force_optimum)


def brute_force(D):
    n = D.shape[0]
    best = None
    for perm in permutations(range(1, n)):
        tour = (0,) + perm
        length = A.tour_length(tour, D)
        if best is None or length < best:
            best = length
    return best


def analyse(settings, keep_history=True):
    result = run(settings, keep_history=keep_history)
    inst, D = instance(settings.n, settings.seed)
    optimum = brute_force(D) if settings.n + 1 <= BRUTE_FORCE_MAX_N else float("nan")
    return Analysis(settings, result, inst, D, optimum)


# --- Sweep (wie die Vorgänger-Demos) ---------------------------------------------------------------------------------------------------------


def run_config(param, value, base, seeds=None):
    seeds = C.SWEEP_SEEDS if seeds is None else seeds
    s0 = replace(base, **{param: value})
    gaps = []
    for run_seed in seeds:
        a = analyse(replace(s0, run_seed=run_seed), keep_history=False)
        gaps.append(a.gap)
    return {"gap": float(np.nanmean(gaps))}


def sweep(param, base=None, values=None):
    # eigenes knappes Budget (ants=4, gens=8) auf der kleinen Vergleichsinstanz - beim komfortablen Kopfexperiment-Budget
    # (COMPARISON_ANTS/COMPARISON_GENS) sättigt der Abstand zum Optimum fast überall bei ~0 %, siehe README
    base = Settings(n=C.COMPARISON_N, seed=C.COMPARISON_VEHICLE_SEED, ants=4, gens=8) if base is None else base
    values = C.SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(param, v, base)} for v in values]


# --- Experiment 1: Kopfexperiment - eigenständig gemessene Abweichung vom Brute-Force-Optimum ------------------------------------------------


def comparison_experiment(n=None, seed=None, seeds=None, ants=None, gens=None):
    n = C.COMPARISON_N if n is None else n
    seed = C.COMPARISON_VEHICLE_SEED if seed is None else seed
    seeds = C.COMPARISON_SEEDS if seeds is None else seeds
    ants = C.COMPARISON_ANTS if ants is None else ants
    gens = C.COMPARISON_GENS if gens is None else gens

    inst, D = instance(n, seed)
    optimum = brute_force(D)

    gaps, lengths = [], []
    for run_seed in seeds:
        s = Settings(n=n, seed=seed, ants=ants, gens=gens, run_seed=run_seed)
        r = run(s, keep_history=False)
        lengths.append(r.best_length)
        gaps.append(max(0.0, 100.0 * (r.best_length - optimum) / optimum))    # nie negativ, siehe Analysis.gap
    return {"optimum": optimum, "gap_median": float(np.median(gaps)), "gap_all": gaps, "lengths_all": lengths}


# --- Experiment 2: eigener Regler - Verdunstungsrate rho --------------------------------------------------------------------------------------


def rho_experiment(n=None, seed=None, values=None, seeds=None, ants=None, gens=None):
    n = C.DEFAULT_N if n is None else n
    seed = C.DEFAULT_SEED if seed is None else seed
    values = C.RHO_VALUES if values is None else values
    seeds = C.RHO_EXPERIMENT_SEEDS if seeds is None else seeds
    ants = C.RHO_EXPERIMENT_ANTS if ants is None else ants
    gens = C.RHO_EXPERIMENT_GENS if gens is None else gens

    rows = []
    for rho in values:
        lengths = []
        for run_seed in seeds:
            s = Settings(n=n, seed=seed, ants=ants, gens=gens, rho=rho, run_seed=run_seed)
            r = run(s, keep_history=False)
            lengths.append(r.best_length)
        rows.append({"rho": rho, "length_median": float(np.median(lengths))})
    return rows
