"""Ant-System-Kern (Dorigo, 1992): klassisches "Ant-Cycle"-System. Kein Kern der Vorgänger-Demos kopiert - andere
Mechanik: Populationsbewegung über indirekte Kommunikation (Pheromonspuren/Stigmergie), keine Mutation/Kovarianz-
Adaption/Differenzvektor/Geschwindigkeit."""

from dataclasses import dataclass, field

import numpy as np


def dist_matrix(xy):
    diff = xy[:, None, :] - xy[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def tour_length(tour, D):
    idx = np.asarray(tour)
    return float(D[idx, np.roll(idx, -1)].sum())


def transition_probs(current, unvisited, tau, eta, alpha, beta):
    """p_ij = tau_ij^alpha * eta_ij^beta / sum_k(tau_ik^alpha * eta_ik^beta), nur über die unbesuchten Knoten."""
    weights = (tau[current, unvisited] ** alpha) * (eta[current, unvisited] ** beta)
    total = weights.sum()
    if total <= 0:
        return np.full(len(unvisited), 1.0 / len(unvisited))
    return weights / total


def construct_tour(tau, eta, alpha, beta, rng, start):
    n = tau.shape[0]
    unvisited = [j for j in range(n) if j != start]
    tour = [start]
    current = start
    while unvisited:
        candidates = np.array(unvisited)
        probs = transition_probs(current, candidates, tau, eta, alpha, beta)
        next_node = int(rng.choice(candidates, p=probs))
        tour.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    return np.array(tour)


def pheromone_update(tau, tours, lengths, rho, q):
    """tau <- (1-rho)*tau + sum_k Δtau^k, Δtau_ij^k = q / L_k für jede Kante (i,j) der Tour von Ameise k."""
    new_tau = (1.0 - rho) * tau
    for tour, length in zip(tours, lengths):
        contribution = q / length
        a = tour
        b = np.roll(tour, -1)
        new_tau[a, b] += contribution
        new_tau[b, a] += contribution
    return new_tau


@dataclass
class Generation:
    tau: np.ndarray
    tours: np.ndarray
    lengths: np.ndarray
    best_tour: np.ndarray


@dataclass
class ACOResult:
    best_tour: np.ndarray
    best_length: float
    best_history: np.ndarray       # (generations,) - bester bisher gefundener Wert NACH Generation g (1-indexiert)
    generations: list = field(default_factory=list)   # nur befüllt, wenn keep_history=True


def run_aco(D, n_ants, generations, alpha, beta, rho, q, tau0, seed, keep_history=False):
    """Ein Ant-System-Lauf. `D`: Distanzmatrix (n_nodes, n_nodes)."""
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    eta = np.zeros_like(D)
    mask = D > 0
    eta[mask] = 1.0 / D[mask]
    tau = np.full((n, n), tau0)
    np.fill_diagonal(tau, 0.0)

    best_tour = None
    best_length = float("inf")
    best_history = []
    gens_snapshots = []

    for _ in range(generations):
        starts = rng.integers(0, n, size=n_ants)
        tours = np.array([construct_tour(tau, eta, alpha, beta, rng, start=int(s)) for s in starts])
        lengths = np.array([tour_length(t, D) for t in tours])

        gen_best_idx = int(np.argmin(lengths))
        if lengths[gen_best_idx] < best_length:
            best_length = float(lengths[gen_best_idx])
            best_tour = tours[gen_best_idx].copy()

        tau = pheromone_update(tau, tours, lengths, rho, q)

        best_history.append(best_length)
        if keep_history:
            gens_snapshots.append(Generation(tau.copy(), tours.copy(), lengths.copy(), best_tour.copy()))

    return ACOResult(best_tour, best_length, np.array(best_history), gens_snapshots)
