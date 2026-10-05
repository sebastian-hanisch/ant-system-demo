"""Orakel-Tests (unabhängiger Rechenweg, ohne neue Abhängigkeiten): Held-Karp-DP statt Permutations-Brute-Force,
Übergangswahrscheinlichkeit/Pheromon-Update als reine Python-Schleifen, Nachrechnung jeder Generation eines Laufs aus
den aufgezeichneten Touren (Längen, Verdunstung + Ablage, Bestwert-Verlauf) und Häufigkeiten der Tourkonstruktion
gegen die exakten Schrittwahrscheinlichkeiten."""

import itertools
import math

import numpy as np
import pytest

import aco_algorithm as A
import aco_evaluation as E


def _points(rng, n):
    return rng.random((n, 2)) * 100


def _dist(xy):
    n = len(xy)
    return [[math.hypot(xy[i][0] - xy[j][0], xy[i][1] - xy[j][1]) for j in range(n)] for i in range(n)]


def held_karp(D):
    n = len(D)
    dp = {(1 << j, j): D[0][j] for j in range(1, n)}
    for size in range(2, n):
        for sub in itertools.combinations(range(1, n), size):
            mask = sum(1 << j for j in sub)
            for j in sub:
                dp[(mask, j)] = min(dp[(mask ^ (1 << j), k)] + D[k][j] for k in sub if k != j)
    full = sum(1 << j for j in range(1, n))
    return min(dp[(full, j)] + D[j][0] for j in range(1, n))


def test_brute_force_reference_matches_held_karp():
    rng = np.random.default_rng(1)
    for _ in range(40):
        n = int(rng.integers(3, 8))
        xy = _points(rng, n)
        assert E.brute_force(A.dist_matrix(xy)) == pytest.approx(held_karp(_dist(xy)), abs=1e-9)


def test_transition_probs_match_plain_python_loops():
    rng = np.random.default_rng(2)
    for _ in range(100):
        n = int(rng.integers(3, 9))
        tau = rng.random((n, n)) * 3 + 0.01
        eta = rng.random((n, n)) * 2 + 0.01
        alpha = float(rng.choice([0.0, 1.0, 2.5]))
        beta = float(rng.choice([0.0, 3.0, 8.0]))
        cur = int(rng.integers(0, n))
        unv = sorted(rng.choice([j for j in range(n) if j != cur], size=int(rng.integers(1, n)), replace=False).tolist())
        w = [math.pow(tau[cur][j], alpha) * math.pow(eta[cur][j], beta) for j in unv]
        got = A.transition_probs(cur, np.array(unv), tau, eta, alpha, beta)
        assert got == pytest.approx([x / sum(w) for x in w], abs=1e-12)


def test_pheromone_update_matches_edge_loop():
    rng = np.random.default_rng(3)
    for _ in range(100):
        n = int(rng.integers(3, 9))
        m = int(rng.integers(1, 6))
        tau = rng.random((n, n))
        tau = (tau + tau.T) / 2
        np.fill_diagonal(tau, 0)
        tours = np.array([rng.permutation(n) for _ in range(m)])
        lengths = rng.random(m) * 100 + 1
        rho = float(rng.choice([0.0, 0.05, 0.5, 1.0]))
        ref = [[(1 - rho) * tau[i][j] for j in range(n)] for i in range(n)]
        for k in range(m):
            for s in range(n):
                a, b = int(tours[k][s]), int(tours[k][(s + 1) % n])
                ref[a][b] += 1.0 / lengths[k]
                ref[b][a] += 1.0 / lengths[k]
        assert A.pheromone_update(tau, tours, lengths, rho, 1.0) == pytest.approx(np.array(ref), abs=1e-12)


def test_recorded_generations_are_consistent_with_independent_recomputation():
    rng = np.random.default_rng(4)
    for seed in range(12):
        n = int(rng.integers(4, 9))
        xy = _points(rng, n)
        Dref = _dist(xy)
        rho = float(rng.choice([0.05, 0.5, 0.95]))
        r = A.run_aco(A.dist_matrix(xy), 5, 4, 1.0, 3.0, rho, 1.0, 1.0, seed, keep_history=True)
        tau = [[0.0 if i == j else 1.0 for j in range(n)] for i in range(n)]
        best = float("inf")
        for g, hist in zip(r.generations, r.best_history):
            lens = []
            for t in g.tours:
                assert sorted(t.tolist()) == list(range(n))
                lens.append(sum(Dref[t[i]][t[(i + 1) % n]] for i in range(n)))
            assert g.lengths == pytest.approx(lens, abs=1e-9)
            best = min(best, min(lens))
            assert hist == pytest.approx(best, abs=1e-9)
            tau = [[(1 - rho) * x for x in row] for row in tau]
            for t, length in zip(g.tours, lens):
                for i in range(n):
                    a, b = int(t[i]), int(t[(i + 1) % n])
                    tau[a][b] += 1.0 / length
                    tau[b][a] += 1.0 / length
            assert g.tau == pytest.approx(np.array(tau), abs=1e-10)
        length = sum(Dref[r.best_tour[i]][r.best_tour[(i + 1) % n]] for i in range(n))
        assert r.best_length == pytest.approx(length, abs=1e-9)


def test_construct_tour_step_frequencies_match_exact_probabilities():
    n = 5
    r = np.random.default_rng(5)
    tau = r.random((n, n)) * 2 + 0.1
    tau = (tau + tau.T) / 2
    np.fill_diagonal(tau, 0)
    D = np.array(_dist(_points(r, n)))
    eta = np.zeros_like(D)
    eta[D > 0] = 1 / D[D > 0]
    alpha, beta, start, N = 1.3, 2.0, 0, 4000
    cnt = {}
    for _ in range(N):
        t = A.construct_tour(tau, eta, alpha, beta, r, start)
        assert sorted(t.tolist()) == list(range(n))
        cnt[(int(t[1]), int(t[2]))] = cnt.get((int(t[1]), int(t[2])), 0) + 1
    w1 = {j: tau[start][j] ** alpha * eta[start][j] ** beta for j in range(1, n)}
    for j in w1:
        rest = [k for k in range(1, n) if k != j]
        w2 = {k: tau[j][k] ** alpha * eta[j][k] ** beta for k in rest}
        for k in rest:
            p = w1[j] / sum(w1.values()) * w2[k] / sum(w2.values())
            assert abs(cnt.get((j, k), 0) / N - p) < 6 * math.sqrt(p * (1 - p) / N) + 2e-3
