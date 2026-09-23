"""Plotly-Abbildungen der Ant-System-Demo: Karte mit pheromonstärke-gewichteten Kanten (wachsendes Beispiel),
Vergleichs- und Sweep-Abbildungen. Achsen sind gesperrt (fixedrange)."""

import numpy as np
import plotly.graph_objects as go

import aco_constants as C

STOP_COLOR = "#4c78a8"
BEST_COLOR = "#54a24b"
PHEROMONE_COLOR = "#e45756"
ACO_COLOR = "#4c78a8"
REF_COLOR = "#7f7f7f"

PHEROMONE_PERCENTILE = 80   # nur Kanten oberhalb dieses Perzentils der aktuellen Pheromonwerte werden gezeichnet (sonst zu unübersichtlich)


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.1), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _map_layout(fig, height=430):
    fig.update_xaxes(range=[-3, C.AREA + 3], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=[-3, C.AREA + 3], showgrid=False, zeroline=False, showticklabels=False)
    return _base(fig, height)


def _tour_edges_list(tour):
    t = np.asarray(tour)
    return list(zip(t.tolist(), np.roll(t, -1).tolist()))


def build_pheromone_map(xy, generation, title=None):
    """Karte mit den Kanten mit der stärksten Pheromonkonzentration (oberhalb PHEROMONE_PERCENTILE, sonst zu
    unübersichtlich) - Linienbreite/-deckkraft proportional zur normierten Pheromonstärke - plus der besten bisher
    gefundenen Tour hervorgehoben."""
    fig = go.Figure()
    tau = generation.tau
    n = tau.shape[0]
    iu = np.triu_indices(n, 1)
    values = tau[iu]
    if len(values) and values.max() > 0:
        threshold = np.percentile(values, PHEROMONE_PERCENTILE)
        vmax = values.max()
        for a, b, v in zip(iu[0], iu[1], values):
            if v < threshold or vmax <= 0:
                continue
            frac = float(v / vmax)
            fig.add_trace(go.Scatter(
                x=[xy[a, 0], xy[b, 0]], y=[xy[a, 1], xy[b, 1]], mode="lines",
                line=dict(color=PHEROMONE_COLOR, width=1 + 5 * frac), opacity=max(0.15, frac),
                showlegend=False, hoverinfo="skip",
            ))

    best_edges = _tour_edges_list(generation.best_tour)
    x, y = [], []
    for a, b in best_edges:
        x += [xy[a, 0], xy[b, 0], None]
        y += [xy[a, 1], xy[b, 1], None]
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=BEST_COLOR, width=2.5, dash="dot"), name="beste Tour bisher"))

    fig.add_trace(go.Scatter(x=xy[1:, 0], y=xy[1:, 1], mode="markers", marker=dict(size=7, color=STOP_COLOR, line=dict(width=1, color="white")), name="Stopps"))
    fig.add_trace(go.Scatter(x=[xy[0, 0]], y=[xy[0, 1]], mode="markers", marker=dict(size=14, symbol="star", color="#f58518", line=dict(width=1, color="white")), name="Depot"))
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=13), x=0.02, y=0.98))
    return _map_layout(fig)


def build_best_curve(best_history, reference=None):
    xs = list(range(1, len(best_history) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=best_history, mode="lines", line=dict(color=ACO_COLOR, width=2.5), name="beste Tour bisher"))
    if reference is not None and np.isfinite(reference):
        fig.add_hline(y=reference, line=dict(color=REF_COLOR, dash="dot"), annotation_text="Brute-Force-Optimum", annotation_position="bottom right")
    fig.update_xaxes(title_text="Generation")
    fig.update_yaxes(title_text="Tourlänge")
    return _base(fig, 260)


def build_comparison(report):
    fig = go.Figure()
    fig.add_trace(go.Box(y=report["gap_all"], name="Abstand zum Optimum", marker_color=ACO_COLOR, boxpoints="all", showlegend=False))
    fig.add_hline(y=0, line=dict(color=REF_COLOR, dash="dot"), annotation_text="Optimum getroffen", annotation_position="top right")
    fig.update_yaxes(title_text="Abstand zum Brute-Force-Optimum (%)")
    return _base(fig, 360)


def build_rho_experiment(rows):
    xs = [r["rho"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["length_median"] for r in rows], mode="lines+markers", line=dict(color=ACO_COLOR, width=2.5), showlegend=False))
    fig.update_xaxes(title_text="Verdunstungsrate ρ")
    fig.update_yaxes(title_text="Tourlänge (Median)")
    return _base(fig, 300)


def build_sweep(rows, param_label):
    xs = [r["value"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["gap"] for r in rows], mode="lines+markers", line=dict(color=ACO_COLOR, width=2.5), showlegend=False))
    fig.update_xaxes(title_text=param_label)
    fig.update_yaxes(title_text="Abstand zum Brute-Force-Optimum (%)")
    return _base(fig, 300)
