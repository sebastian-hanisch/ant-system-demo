"""Ant System - Stigmergie statt Population-Operatoren - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Neuntes Stück der Populations-Metaheuristiken-Linie der "Konzepte"-Reihe, vierter und letzter unabhängiger KONTRAST
zu GA (kein Fix): Ant System (Dorigo, 1992) bewegt eine Population über INDIREKTE Kommunikation - Pheromonspuren auf
den Kanten des Graphen (Stigmergie) - statt Mutation/Kovarianz-Adaption/Differenzvektor/Geschwindigkeit. Vehikel ist
dieselbe diskrete Lieferroute wie genetic-algorithm-demo/nsga2-demo/nsga3-demo/moead-demo - das klassische TSP-Setting,
für das ACO ursprünglich erfunden wurde.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import aco_constants as C
from aco_evaluation import Settings, analyse, comparison_experiment, rho_experiment, sweep
from aco_presets import apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_run_seed, randomize_seed, sync_query_params
from aco_visualization import build_best_curve, build_comparison, build_pheromone_map, build_rho_experiment, build_sweep

st.set_page_config(page_title="Ant System – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings, keep_history=True)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _comparison():
    return comparison_experiment()


@st.cache_data(show_spinner=False)
def _rho_experiment():
    return rho_experiment()


st.title("🐜 Ant System – Stigmergie statt Population-Operatoren")
st.markdown(
    """
GA kombiniert per Crossover, CMA-ES/DE/L-SHADE/PSO bewegen sich über Mutation/Kovarianz-Adaption/Geschwindigkeit auf
kontinuierlichen Landschaften. **Ant System** (Dorigo, 1992) geht für diskrete Probleme wie die Tourenplanung einen
ganz anderen Weg: Ameisen kommunizieren nicht direkt miteinander, sondern **indirekt über Pheromonspuren** auf den
Straßenabschnitten (Stigmergie) - jede Ameise baut eine komplette Tour, bevorzugt stark befahrene (pheromonreiche) UND
kurze Abschnitte, und hinterlässt danach selbst mehr Pheromon auf kürzeren Touren. Über viele Generationen bilden sich
so "heiße" Pfade heraus.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - "
    "neuntes Stück der Populations-Metaheuristiken-Linie der \"Konzepte\"-Reihe, ein **Kontrast** zu "
    "[genetic-algorithm-demo](https://sebastianhanisch-genetic-algorithm-demo.streamlit.app/) statt eines Fixes - "
    "**ein** Verfahren an einem wachsenden Beispiel. Vehikel ist dieselbe diskrete Lieferroute wie dort/bei "
    "[nsga2-demo](https://sebastianhanisch-nsga2-demo.streamlit.app/)."
)

with st.expander("So funktioniert Ant System", expanded=True):
    st.markdown(
        r"""
1. **Tourkonstruktion.** Jede Ameise baut, Knoten für Knoten, eine vollständige Tour. Die Übergangswahrscheinlichkeit
   zum nächsten Stopp $j$ ist $p_{ij} = \dfrac{\tau_{ij}^\alpha \eta_{ij}^\beta}{\sum_k \tau_{ik}^\alpha \eta_{ik}^\beta}$ -
   $\tau$ = Pheromonstärke, $\eta = 1/\text{Distanz}$ (Sichtbarkeit).
2. **Verdunstung.** Nach jeder Generation verliert jede Kante einen Anteil $\rho$ ihres Pheromons.
3. **Ablage.** Jede Ameise hinterlässt auf den Kanten IHRER Tour Pheromon proportional zu $1/\text{Tourlänge}$ - kürzere
   Touren hinterlassen mehr.
4. **Indirekte Kommunikation.** Keine Ameise "sieht" eine andere - alle Koordination läuft über die gemeinsam
   veränderte Umgebung (die Pheromonspur selbst).
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS.keys())
cols = st.columns(len(preset_names))
for col, name in zip(cols, preset_names):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name], key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_stops = st.slider("Stopps", *bounds("n_slider"), key="n_slider", step=C.N_STEP, help="Anzahl der Kundenstopps (das Depot kommt dazu).")
    st.markdown("**Ant System**")
    ants = st.slider("Ameisenzahl", *bounds("ants_slider"), key="ants_slider", step=C.ANTS_STEP)
    generations = st.slider("Generationen", *bounds("gens_slider"), key="gens_slider", step=C.GEN_STEP)
    alpha = st.slider("Pheromon-Exponent α", *bounds("alpha_slider"), key="alpha_slider", step=C.ALPHA_STEP, help="Wie stark Pheromon die Wegwahl beeinflusst.")
    beta = st.slider("Sichtbarkeits-Exponent β", *bounds("beta_slider"), key="beta_slider", step=C.BETA_STEP, help="Wie stark die Distanz (1/Länge) die Wegwahl beeinflusst.")
    rho = st.slider("Verdunstungsrate ρ", *bounds("rho_slider"), key="rho_slider", step=C.RHO_STEP, help="Welcher Anteil des Pheromons je Generation verdunstet.")
    seed = st.number_input("Zufalls-Seed des Vehikels", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neues Vehikel generieren", width="stretch", on_click=randomize_seed)
    run_seed = st.number_input("Zufalls-Seed des Ant-System-Laufs", *bounds("run_seed_input"), key="run_seed_input", step=1)
    st.button("🎲 Neuen Lauf würfeln", width="stretch", on_click=randomize_run_seed)

sync_query_params({
    "n_slider": int(n_stops), "ants_slider": int(ants), "gens_slider": int(generations),
    "alpha_slider": float(alpha), "beta_slider": float(beta), "rho_slider": float(rho),
    "seed_input": int(seed), "run_seed_input": int(run_seed),
})

settings = Settings(n=int(n_stops), seed=int(seed), ants=int(ants), gens=int(generations), alpha=float(alpha), beta=float(beta), rho=float(rho), run_seed=int(run_seed))
with st.spinner("Rechne..."):
    a = _analysis(settings)
result = a.result
n_gens_run = len(result.generations)
data_key = settings

# --- Ant System in Aktion ----------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ant System in Aktion")
if "aco_gen" not in st.session_state or st.session_state.get("aco_gen_owner") != data_key:
    st.session_state["aco_gen"] = n_gens_run
    st.session_state["aco_gen_owner"] = data_key
gen_col, play_col = st.columns([5, 2])
with gen_col:
    gen = st.slider("Generation", 1, n_gens_run, key="aco_gen")
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
view_slot = st.empty()


def _frames():
    return sorted({int(round(x)) for x in np.linspace(1, n_gens_run, min(n_gens_run, 40))})


def _render(g):
    gd = result.generations[g - 1]
    with view_slot.container():
        c1, c2 = st.columns([3, 2])
        c1.markdown(f"**Generation {g} von {n_gens_run} – beste Tour bisher: {result.best_history[g - 1]:.1f} km**")
        c1.plotly_chart(build_pheromone_map(a.inst.xy, gd), width="stretch", key=f"g_map_{g}")
        c2.markdown("**Beste Tour bisher**")
        c2.plotly_chart(build_best_curve(result.best_history[:g], reference=a.brute_force_optimum), width="stretch", key=f"g_best_{g}")


if auto_play:
    for fr in _frames():
        _render(fr)
        time.sleep(0.15)
else:
    _render(gen)

st.markdown("---")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was Ant System gefunden hat")
m1, m2 = st.columns(2)
m1.metric("Beste gefundene Tour", f"{result.best_length:.1f} km")
if np.isfinite(a.brute_force_optimum):
    m2.metric("Abstand zum Brute-Force-Optimum", f"{a.gap:+.1f} %")
else:
    m2.metric("Brute-Force-Optimum", "nicht berechenbar (zu viele Stopps)")

st.markdown("---")

# --- Sweep -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt der Abstand zum Optimum von Ameisenzahl und Verdunstungsrate ab?")
st.caption("Läuft auf der kleinen Vergleichsinstanz (8 Stopps) mit eigenem knappen Budget - beim komfortablen Kopfexperiment-Budget trifft fast jede Einstellung das Optimum.")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(C.SWEEP_LABELS), format_func=lambda k: C.SWEEP_LABELS[k], key="sweep_select")
if st.button("Sweep über 5 feste Vehikel berechnen (dauert etwa 10 bis 30 Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {sweep_param}
if sweep_param in st.session_state.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep..."):
        rows_sweep = _sweep(sweep_param, None)
    st.plotly_chart(build_sweep(rows_sweep, C.SWEEP_LABELS[sweep_param]), width="stretch", key="sweep_chart")

st.markdown("---")

# --- Experiment 1: Kopfexperiment - eigenständig gemessene Abweichung vom Optimum --------------------------------------------------------------

st.subheader("🔬 Wie nah kommt Ant System ans echte Optimum?")
st.caption(f"Kleine Vergleichsinstanz ({C.COMPARISON_N} Stopps, Vehikel-Seed {C.COMPARISON_VEHICLE_SEED}, Brute-Force-lösbar) - Abstand zum echten Optimum über {len(C.COMPARISON_SEEDS)} Läufe.")
if st.button("Brute-Force-Optimum gegen Ant System rechnen (dauert etwa 5 Sekunden)", key="comparison_start"):
    st.session_state["comparison_on"] = True
if st.session_state.get("comparison_on"):
    with st.spinner("Rechne die Brute-Force-Front und mehrere Ant-System-Läufe..."):
        report = _comparison()
    st.plotly_chart(build_comparison(report), width="stretch", key="comparison_chart")
    st.metric("Abstand zum Optimum, Median über 20 Läufe", f"{report['gap_median']:+.1f} %")

st.markdown("---")

# --- Experiment 2: eigener Regler - Verdunstungsrate rho --------------------------------------------------------------------------------------

st.subheader("🔬 Wie stark hängt die Tourqualität von der Verdunstungsrate ρ ab?")
st.caption("Zu klein: altes Pheromon verschwindet nie, frühe Pfade dominieren dauerhaft. Zu groß: die Kolonie vergisst gute Pfade, bevor sie sich verstärken können.")
if st.button(f"Verdunstungsraten {C.RHO_VALUES[0]:.2f} bis {C.RHO_VALUES[-1]:.2f} vergleichen (dauert etwa 10 Sekunden)", key="rho_start"):
    st.session_state["rho_on"] = True
if st.session_state.get("rho_on"):
    with st.spinner("Rechne 5 Verdunstungsraten × 20 Läufe..."):
        rows_rho = _rho_experiment()
    st.plotly_chart(build_rho_experiment(rows_rho), width="stretch", key="rho_chart")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Keine Pheromon-Obergrenze** | Ein früh gefundener, mittelmäßiger Pfad kann sich unbegrenzt verstärken - die ganze Kolonie legt sich vorzeitig darauf fest (Stagnation), ohne Mechanismus, das zu erkennen oder zu korrigieren. | Max-Min Ant System (τ_min/τ_max-Schranken) - separates, geplantes Folgestück |
| **Verdunstungsrate ρ ist gut gewählt** | Zu klein: altes Pheromon dominiert dauerhaft. Zu groß: gute Pfade verschwinden, bevor sie sich verstärken. | Muss von Hand eingestellt werden, wie bei jedem Regler dieser Linie |
| **Einfache Pheromon-Initialisierung** | τ0=1 (konstant) statt der literaturüblichen NN-Tour-basierten Initialisierung ($\\tau_0 = 1/(n \\cdot L_{NN})$) - bewusste Vereinfachung. | Hier nicht umgesetzt, siehe README |
| **Kein Distanz-basierter Suchraum-Vorteil bei sehr großen Instanzen** | Bei vielen Stopps wächst der Rechenaufwand pro Generation mit $O(\\text{Ameisen} \\times \\text{Stopps}^2)$ - keine Beschleunigungstechniken (Kandidatenlisten etc.) umgesetzt. | Hier nicht umgesetzt |
"""
)
st.caption(
    "Ant System ist der vierte und letzte unabhängige Kontrast-Ast von GA in dieser Linie. Der geplante Fix-Nachfolger "
    "Max-Min Ant System (behebt die Stagnationsgefahr) ist ein separates, noch nicht gebautes Stück."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Übergangswahrscheinlichkeit.** $p_{ij} = \dfrac{\tau_{ij}^\alpha \eta_{ij}^\beta}{\sum_{k \in \text{erlaubt}}
\tau_{ik}^\alpha \eta_{ik}^\beta}$, mit $\eta_{ij} = 1/d_{ij}$.

**Verdunstung + Ablage** ("Ant-Cycle"-System). $\tau_{ij} \leftarrow (1-\rho)\tau_{ij} + \sum_k \Delta\tau_{ij}^k$,
mit $\Delta\tau_{ij}^k = Q / L_k$ falls Kante $(i,j)$ in der Tour von Ameise $k$ vorkommt (Tourlänge $L_k$), sonst 0.

Implementiert in `aco_algorithm.py` (Übergangswahrscheinlichkeit, Tourkonstruktion, Pheromon-Update, Hauptschleife),
`aco_scenario.py` (Vehikel), `aco_evaluation.py` (Kennzahlen, Sweep, Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
