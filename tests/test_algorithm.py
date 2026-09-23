"""Handrechnungen für Übergangswahrscheinlichkeit/Pheromon-Update, Brute-Force-Vergleich auf einer sehr kleinen
Instanz - eigene Implementierung UND `acopy` (verbreitetes ACO-Referenzpaket, Quelltext bestätigt: identische
Verdunstungs-/Ablage-Formel τ <- (1-ρ)τ + Σ q/L) im Vergleich."""

from itertools import permutations

import networkx as nx
import numpy as np
import pytest
from acopy import Colony, Solver

import aco_algorithm as A


def test_transition_probs_matches_hand_calculation():
    tau = np.array([[0.0, 2.0, 3.0], [2.0, 0.0, 1.0], [3.0, 1.0, 0.0]])
    eta = np.array([[0.0, 0.5, 0.25], [0.5, 0.0, 1.0], [0.25, 1.0, 0.0]])
    probs = A.transition_probs(0, np.array([1, 2]), tau, eta, alpha=1, beta=1)
    # weights = [2*0.5, 3*0.25] = [1.0, 0.75] -> normiert [0.571..., 0.428...]
    assert probs == pytest.approx([1.0 / 1.75, 0.75 / 1.75])


def test_transition_probs_ignores_pheromone_when_alpha_is_zero():
    tau = np.array([[0.0, 100.0, 1.0], [100.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    eta = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 2.0], [2.0, 2.0, 0.0]])
    probs = A.transition_probs(0, np.array([1, 2]), tau, eta, alpha=0.0, beta=1.0)
    assert probs == pytest.approx([1.0 / 3.0, 2.0 / 3.0])   # nur noch eta zaehlt


def test_pheromone_update_matches_hand_calculation():
    tau = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    tours = np.array([[0, 1, 2]])
    lengths = np.array([10.0])
    new_tau = A.pheromone_update(tau, tours, lengths, rho=0.5, q=1.0)
    expected = np.array([[0.0, 0.6, 0.6], [0.6, 0.0, 0.6], [0.6, 0.6, 0.0]])
    assert new_tau == pytest.approx(expected)


def test_pheromone_update_sums_contributions_from_multiple_ants():
    tau = np.zeros((3, 3))
    tours = np.array([[0, 1, 2], [0, 2, 1]])
    lengths = np.array([10.0, 5.0])
    new_tau = A.pheromone_update(tau, tours, lengths, rho=0.0, q=1.0)
    # Kante (0,1): Ameise 1 nutzt sie (0->1), Ameise 2 nicht (Tour 0->2->1 nutzt (1,0) via Rueckkehr... pruefen wir explizit)
    assert new_tau[0, 1] == pytest.approx(1.0 / 10.0 + 1.0 / 5.0)   # beide Touren durchlaufen die Kante (0,1) in einer Richtung


def test_construct_tour_visits_every_node_exactly_once():
    rng = np.random.default_rng(1)
    n = 6
    tau = np.ones((n, n))
    eta = np.ones((n, n))
    np.fill_diagonal(tau, 0.0)
    np.fill_diagonal(eta, 0.0)
    for start in range(n):
        tour = A.construct_tour(tau, eta, alpha=1.0, beta=1.0, rng=rng, start=start)
        assert sorted(tour.tolist()) == list(range(n))
        assert tour[0] == start


def test_tour_length_matches_manual_computation():
    xy = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
    D = A.dist_matrix(xy)
    length = A.tour_length([0, 1, 2], D)
    assert length == pytest.approx(3.0 + 4.0 + 5.0)   # 0->1: 3, 1->2: 4, 2->0: 5 (3-4-5-Dreieck)


# --- Brute-Force-Vergleich auf einer sehr kleinen Instanz -------------------------------------------------------------------------------------


def brute_force_optimum(D):
    n = D.shape[0]
    best = None
    for perm in permutations(range(1, n)):
        tour = (0,) + perm
        length = A.tour_length(tour, D)
        if best is None or length < best:
            best = length
    return best


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_run_aco_finds_the_brute_force_optimum_on_a_tiny_instance(seed):
    rng = np.random.default_rng(seed)
    n = 6
    xy = rng.random((n, 2)) * 100
    D = A.dist_matrix(xy)
    optimum = brute_force_optimum(D)

    r = A.run_aco(D, n_ants=15, generations=40, alpha=1.0, beta=3.0, rho=0.3, q=1.0, tau0=1.0, seed=seed)
    assert r.best_length == pytest.approx(optimum, rel=0.05)   # nahe am Optimum, kein exaktes Gleichstand-Erfordernis


def test_run_aco_and_acopy_reach_a_comparably_good_tour():
    """Kein Generation-für-Generation-Gleichlauf (acopy startet Pheromon bei 0, verwendet Zufallsstarts anders) -
    aber beide Implementierungen sollen auf derselben kleinen Instanz nahe am Brute-Force-Optimum landen."""
    rng = np.random.default_rng(7)
    n = 7
    xy = rng.random((n, 2)) * 100
    D = A.dist_matrix(xy)
    optimum = brute_force_optimum(D)

    r = A.run_aco(D, n_ants=20, generations=60, alpha=1.0, beta=3.0, rho=0.3, q=1.0, tau0=1.0, seed=1)

    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=float(D[i, j]))
    solver = Solver(rho=0.3, q=1)
    colony = Colony(alpha=1, beta=3)
    solution = solver.solve(G, colony, limit=60, gen_size=20)

    assert r.best_length == pytest.approx(optimum, rel=0.05)
    assert solution.cost == pytest.approx(optimum, rel=0.05)


def test_run_aco_history_shapes_and_generations_snapshots():
    rng = np.random.default_rng(1)
    n = 6
    xy = rng.random((n, 2)) * 100
    D = A.dist_matrix(xy)
    gens = 10
    r = A.run_aco(D, n_ants=8, generations=gens, alpha=1.0, beta=3.0, rho=0.3, q=1.0, tau0=1.0, seed=1, keep_history=True)
    assert r.best_history.shape == (gens,)
    assert len(r.generations) == gens
    for g in r.generations:
        assert g.tau.shape == (n, n)
        assert g.tours.shape == (8, n)
        assert g.lengths.shape == (8,)


def test_run_aco_best_history_is_monotonically_non_increasing():
    rng = np.random.default_rng(1)
    n = 8
    xy = rng.random((n, 2)) * 100
    D = A.dist_matrix(xy)
    r = A.run_aco(D, n_ants=10, generations=30, alpha=1.0, beta=3.0, rho=0.3, q=1.0, tau0=1.0, seed=1)
    assert np.all(np.diff(r.best_history) <= 1e-9)


def test_run_aco_without_history_leaves_generations_empty():
    rng = np.random.default_rng(1)
    xy = rng.random((5, 2)) * 100
    D = A.dist_matrix(xy)
    r = A.run_aco(D, n_ants=5, generations=5, alpha=1.0, beta=3.0, rho=0.3, q=1.0, tau0=1.0, seed=1, keep_history=False)
    assert r.generations == []
