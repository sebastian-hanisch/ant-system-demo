"""Vehikel (Reproduzierbarkeit, bitidentisch zu genetic-algorithm-demo/nsga2-demo) und Auswertung (Distanzmatrix,
Brute-Force, Sweep, beide Experimente) - schnelle Parameter über Funktionsargumente."""

import numpy as np
import pytest

import aco_algorithm as A
import aco_constants as C
import aco_evaluation as E
import aco_scenario as S


def test_generate_perm_is_reproducible_and_shaped():
    a = S.generate_perm(20, 0, seed=7)
    b = S.generate_perm(20, 0, seed=7)
    assert np.array_equal(a.xy, b.xy)
    assert a.xy.shape == (21, 2)


def test_generate_perm_matches_genetic_algorithm_demo_bit_for_bit_on_the_comparison_instance():
    """Die kleine Vergleichsinstanz (n=8, Seed 19) muss xy bitidentisch zu genetic-algorithm-demo/nsga2-demo liefern -
    reproduziert deren generate_perm hier lokal (kein Cross-Repo-Import, wie überall im Portfolio)."""
    def ref_generate_perm(n, cluster_share, seed):
        rng = np.random.default_rng(seed)
        n_grouped = int(round(n * cluster_share / 100))
        uniform = rng.random((n - n_grouped, 2)) * C.AREA
        centres = C.CLUSTER_MARGIN + rng.random((C.N_CLUSTERS, 2)) * (C.AREA - 2 * C.CLUSTER_MARGIN)
        which = rng.integers(0, C.N_CLUSTERS, size=n_grouped)
        grouped = np.clip(centres[which] + rng.normal(0.0, C.CLUSTER_SIGMA, size=(n_grouped, 2)), 0.0, C.AREA)
        depot = np.array([[C.AREA / 2, C.AREA / 2]])
        return np.vstack([depot, uniform, grouped])

    xy_ref = ref_generate_perm(C.COMPARISON_N, 0, C.COMPARISON_VEHICLE_SEED)
    inst = S.generate_perm(C.COMPARISON_N, 0, C.COMPARISON_VEHICLE_SEED)
    assert np.array_equal(inst.xy, xy_ref)


# --- Auswertung --------------------------------------------------------------------------------------------------------------------------------


def test_run_returns_expected_shapes():
    s = E.Settings(n=10, gens=10, ants=6)
    r = E.run(s, keep_history=True)
    assert r.best_history.shape == (10,)
    assert len(r.generations) == 10


def test_brute_force_matches_manual_computation_on_a_tiny_instance():
    inst, D = E.instance(5, 3)
    optimum = E.brute_force(D)
    # unabhängige Nachrechnung über itertools.permutations wie in test_algorithm.py
    from itertools import permutations
    best = min(A.tour_length((0,) + p, D) for p in permutations(range(1, 6)))
    assert optimum == pytest.approx(best)


def test_analyse_computes_gap_only_within_brute_force_range():
    a_small = E.analyse(E.Settings(n=C.COMPARISON_N, seed=C.COMPARISON_VEHICLE_SEED, gens=10))
    assert np.isfinite(a_small.brute_force_optimum)
    assert np.isfinite(a_small.gap)

    a_large = E.analyse(E.Settings(n=50, gens=5))
    assert not np.isfinite(a_large.brute_force_optimum)
    assert not np.isfinite(a_large.gap)


def test_sweep_smoke():
    rows = E.sweep("ants", base=E.Settings(n=C.COMPARISON_N, seed=C.COMPARISON_VEHICLE_SEED, ants=4, gens=6), values=(4, 10))
    assert len(rows) == 2
    assert all(np.isfinite(r["gap"]) for r in rows)


def test_comparison_experiment_smoke_small():
    report = E.comparison_experiment(seeds=(1, 2))
    assert report["optimum"] > 0
    assert np.isfinite(report["gap_median"])
    assert len(report["gap_all"]) == 2


def test_rho_experiment_smoke_small():
    rows = E.rho_experiment(values=(0.05, 0.95), seeds=(1, 2), ants=6, gens=10)
    assert len(rows) == 2
    assert all(r["length_median"] > 0 for r in rows)
