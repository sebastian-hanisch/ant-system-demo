"""Jede im README/PRESET_HELP/App genannte Zahl wird hier nachgerechnet - keine Behauptung ohne Test.

Einzelne 8-80-Generationen-Läufe sind chaotisch empfindlich gegenüber winziger Fließkomma-Rundung (siehe
feedback_ci_platform_robust_tests.md, und die eigene Erfahrung aus nsga2-demo/nsga3-demo/moead-demo/cma-es-demo/
differential-evolution-demo/l-shade-demo/pso-demo). Zahlen aus einem EINZELNEN Lauf (Presets) bekommen deshalb nur
Strukturgrenzen; Zahlen, die über mehrere Seeds mitteln (Experimente, Sweep), sind von Natur aus robuster und dürfen
engere (aber weiterhin großzügige) Bänder bekommen."""

import numpy as np
import pytest

import aco_constants as C
import aco_evaluation as E


def _preset_analysis(name):
    p = C.PRESETS[name]
    s = E.Settings(n=p["n"], seed=p["seed"], ants=p["ants"], gens=p["gens"], alpha=p["alpha"], beta=p["beta"], rho=p["rho"], run_seed=p["run_seed"])
    return E.analyse(s, keep_history=False)


# --- Einzelläufe (Presets) - nur Strukturgrenzen, keine Nähe zu einem Messwert ------------------------------------------------------------


def test_standardfall_preset_claims():
    a = _preset_analysis("Standardfall")
    assert 0.0 < a.result.best_length < 2000.0
    assert not np.isfinite(a.brute_force_optimum)     # 30 Stopps sind nicht brute-force-lösbar


def test_kleine_verdunstungsrate_preset_claims():
    a = _preset_analysis("Kleine Verdunstungsrate")
    assert 0.0 < a.result.best_length < 2000.0


def test_grosse_verdunstungsrate_preset_claims():
    a = _preset_analysis("Große Verdunstungsrate")
    assert 0.0 < a.result.best_length < 2000.0


def test_kleine_instanz_preset_claims():
    a = _preset_analysis("Kleine Instanz (Vergleich mit Brute-Force)")
    assert np.isfinite(a.brute_force_optimum)
    assert -1e-6 <= a.gap < 50.0


# --- Headlinezahlen der beiden Experimente + Sweep (mitteln über mehrere Seeds, robuster) -----------------------------------------------------


def test_comparison_experiment_headline_claims():
    report = E.comparison_experiment()
    assert report["optimum"] > 0
    assert all(g >= 0.0 for g in report["gap_all"])
    # Kernbefund: Ant System trifft die kleine Vergleichsinstanz im Median (fast) exakt
    assert report["gap_median"] < 2.0


def test_rho_experiment_headline_claims():
    rows = E.rho_experiment()
    by_rho = {r["rho"]: r for r in rows}
    assert set(by_rho) == set(C.RHO_VALUES)
    for r in rows:
        assert r["length_median"] > 0.0
    # Kernbefund: die kleinste Verdunstungsrate schneidet klar schlechter ab als der Standardwert
    assert by_rho[C.RHO_VALUES[0]]["length_median"] > by_rho[C.DEFAULT_RHO]["length_median"]


def test_ants_sweep_headline_claims():
    rows = E.sweep("ants")
    assert [r["value"] for r in rows] == list(C.SWEEP_VALUES["ants"])
    for r in rows:
        assert r["gap"] >= 0.0
    # Kernbefund: zu wenige Ameisen schaden klar erkennbar
    assert rows[0]["gap"] > rows[-1]["gap"]
