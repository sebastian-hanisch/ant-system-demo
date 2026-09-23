"""Konstanten der Ant-System-Demo: Vehikel (wie genetic-algorithm-demo, diskrete Lieferroute), Ant-System-Regler,
Presets (Presets folgen nach den Messungen)."""

# --- Vehikel: Lieferroute (wortgleich aus genetic-algorithm-demo/ga_constants.py) --------------------------------------------------------

AREA = 100.0
N_CLUSTERS = 5
CLUSTER_SIGMA = 6.0                # Streuung einer Gruppe in km
CLUSTER_MARGIN = 12.0              # Gruppenmittelpunkte liegen mindestens so weit vom Rand entfernt
N_MIN, N_MAX, DEFAULT_N, N_STEP = 8, 100, 30, 2    # N_MIN=8, N_STEP=2: Slider trifft sowohl COMPARISON_N=8 als auch DEFAULT_N=30 exakt

# --- Ant System --------------------------------------------------------------------------------------------------------------------------

ANTS_MIN, ANTS_MAX, DEFAULT_ANTS, ANTS_STEP = 4, 100, 20, 2
GEN_MIN, GEN_MAX, DEFAULT_GEN, GEN_STEP = 10, 300, 80, 10
ALPHA_MIN, ALPHA_MAX, DEFAULT_ALPHA, ALPHA_STEP = 0.0, 5.0, 1.0, 0.1     # Pheromon-Exponent
BETA_MIN, BETA_MAX, DEFAULT_BETA, BETA_STEP = 0.0, 8.0, 3.0, 0.5        # Sichtbarkeits-Exponent (1/Distanz)
RHO_MIN, RHO_MAX, DEFAULT_RHO, RHO_STEP = 0.05, 0.95, 0.5, 0.05         # Verdunstungsrate
Q = 1.0                            # Ablage-Konstante (Δτ = Q / Tourlänge)
TAU0 = 1.0                         # Pheromon-Startwert (einfache Initialisierung, siehe README)
SEED_MAX = 999999
DEFAULT_SEED = 35                  # Vehikel-Seed (wie genetic-algorithm-demo, bitidentische Lieferroute)
DEFAULT_RUN_SEED = 7               # Seed des Ant-System-Laufs selbst

# --- Kopfexperiment: eigenständig gemessene Abweichung vom Brute-Force-Optimum (kein Cross-Repo-Zitat) --------------------------------------
# Auf der kleinen Vergleichsinstanz (n=8, Seed 19 - wie nsga2-demo/nsga3-demo/moead-demo) ist Brute-Force lösbar
# ((n-1)! = 5040 Touren). ACOs eigene Kennzahl ist der Abstand zum echten Optimum, nicht eine "Trefferquote im
# globalen Trichter" wie in der kontinuierlichen Teil-Linie (CMA-ES/DE/L-SHADE/PSO) - andere Vergleichsgröße, deshalb
# hier kein Zitat aus jenen Demos.

COMPARISON_N = 8
COMPARISON_VEHICLE_SEED = 19
COMPARISON_SEEDS = tuple(range(2500000, 2500020))
COMPARISON_ANTS, COMPARISON_GENS = 20, 80

# --- Eigener Regler: Verdunstungsrate rho -----------------------------------------------------------------------------------------------------

RHO_VALUES = (0.05, 0.2, 0.5, 0.8, 0.95)
RHO_EXPERIMENT_SEEDS = tuple(range(2600000, 2600020))
RHO_EXPERIMENT_ANTS = 10
RHO_EXPERIMENT_GENS = 30

SWEEP_SEEDS = tuple(range(2700000, 2700005))
SWEEP_VALUES = {"ants": (4, 10, 20, 40, 80), "rho": RHO_VALUES}
SWEEP_LABELS = {"ants": "Ameisenzahl", "rho": "Verdunstungsrate ρ"}


def _preset(n=DEFAULT_N, ants=DEFAULT_ANTS, gens=DEFAULT_GEN, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, rho=DEFAULT_RHO, seed=DEFAULT_SEED, run_seed=DEFAULT_RUN_SEED):
    return {"n": n, "ants": ants, "gens": gens, "alpha": alpha, "beta": beta, "rho": rho, "seed": seed, "run_seed": run_seed}


PRESETS = {
    "Standardfall": _preset(),
    "Kleine Verdunstungsrate": _preset(rho=RHO_VALUES[0]),
    "Große Verdunstungsrate": _preset(rho=RHO_VALUES[-1]),
    "Kleine Instanz (Vergleich mit Brute-Force)": _preset(n=COMPARISON_N, seed=COMPARISON_VEHICLE_SEED, ants=COMPARISON_ANTS, gens=COMPARISON_GENS),
}
PRESET_HELP = {
    "Standardfall": "30 Stopps, 20 Ameisen, ρ=0,5, 80 Generationen (Standard-Seed): beste gefundene Tour 486,9 km.",
    "Kleine Verdunstungsrate": "ρ=0,05 statt 0,5: beste Tour 497,0 km - etwas schlechter, altes Pheromon dominiert länger (siehe Regler-Experiment: über mehrere Seeds gemittelt klar am schlechtesten).",
    "Große Verdunstungsrate": "ρ=0,95 statt 0,5: beste Tour 489,2 km - fast gleich gut wie der Standardfall (siehe Regler-Experiment: kaum schlechter als ρ=0,5 über mehrere Seeds).",
    "Kleine Instanz (Vergleich mit Brute-Force)": "8 Stopps, Brute-Force-lösbar: findet das echte Optimum exakt (255,4 km, 0,0 % Abstand) - über 20 Läufe gemittelt trifft Ant System hier fast immer das Optimum (siehe Kopfexperiment).",
}
