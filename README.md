# 🐜 Ant System – Stigmergie statt Population-Operatoren

Neuntes Stück der **Populations-Metaheuristiken-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) –
Operations Research und Machine Learning. Vierter und letzter unabhängiger Kontrast zu [genetic-algorithm-demo](https://sebastianhanisch-genetic-algorithm-demo.streamlit.app/):
Ant System (Dorigo, 1992) bewegt eine Population über **indirekte Kommunikation** - Pheromonspuren auf den Kanten
des Graphen (Stigmergie) - statt Mutation/Kovarianz-Adaption/Differenzvektor/Geschwindigkeit. Vehikel ist dieselbe
diskrete Lieferroute wie genetic-algorithm-demo/nsga2-demo/nsga3-demo/moead-demo - das klassische TSP-Setting, für
das ACO ursprünglich erfunden wurde. Hat einen geplanten Fix-Nachfolger Max-Min Ant System (Stützle & Hoos, 1996)
für ein separates, noch nicht gebautes Stück.

## Warum dieses Problem

GA, NSGA-II, MOEA/D arbeiten alle mit einer expliziten Population, die per Crossover/Mutation weiterentwickelt wird -
jedes Individuum "kennt" implizit die anderen über die gemeinsame Selektion. Ant System geht für Routenprobleme einen
biologisch inspirierten anderen Weg: Ameisen kommunizieren **nicht direkt**. Jede Ameise baut eine komplette Tour,
bevorzugt dabei kurze UND stark pheromonmarkierte Straßenabschnitte, und hinterlässt danach selbst mehr Pheromon auf
kürzeren Touren. Die ganze Koordination läuft über die gemeinsam veränderte Umgebung - genau das nennt man
Stigmergie.

## Modell

Dieselbe diskrete Lieferroute wie genetic-algorithm-demo/nsga2-demo/nsga3-demo/moead-demo: ein Depot in der Mitte und
*n* Kundenstopps in einem 100×100-km-Gebiet. **`aco_scenario.generate_perm` reproduziert die Vehikel-Erzeugung
wortgleich** (nur die xy-Koordinaten - kein CO2-Faktor wie bei den GA-Geschwistern, Ant System ist hier bewusst
einzielig) - bei Standard-Vehikel-Seed 35 bzw. der geteilten kleinen Vergleichsinstanz (n=8, Seed 19) bitidentisch zu
den Vorgänger-Demos.

## Methodik

Klassisches "Ant-Cycle"-Ant-System (`aco_algorithm.py`, kein Kern der Vorgänger-Demos kopiert - andere Mechanik):
Übergangswahrscheinlichkeit $p_{ij} \propto \tau_{ij}^\alpha \eta_{ij}^\beta$ ($\eta$ = 1/Distanz), jede Ameise baut
eine vollständige Tour, danach Verdunstung + Ablage: $\tau_{ij} \leftarrow (1-\rho)\tau_{ij} + \sum_k \Delta\tau_{ij}^k$
mit $\Delta\tau_{ij}^k = Q/L_k$. Pheromon startet bei $\tau_0=1$ (einfache Initialisierung, siehe Grenzen).

**Kreuzprobe gegen `acopy`** (verbreitetes ACO-Referenzpaket - Quelltext bestätigt: identische Verdunstungs-/
Ablage-Formel $\tau \leftarrow (1-\rho)\tau + \sum Q/L$): beide Implementierungen finden auf kleinen,
brute-force-lösbaren Instanzen zuverlässig das exakte Optimum oder liegen sehr nahe daran. Zusätzlich Handrechnungen
für Übergangswahrscheinlichkeit und Pheromon-Update an konstruierten Beispielen.

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| Wie nah kommt Ant System ans echte Optimum? | Auf der kleinen Vergleichsinstanz (8 Stopps, Brute-Force-lösbar) trifft Ant System im Median über 20 Läufe (fast) exakt das Optimum. | `test_comparison_experiment_headline_claims` |
| Wie stark hängt die Tourqualität von der Verdunstungsrate ρ ab? | Eine zu kleine Verdunstungsrate (ρ=0,05) schneidet über mehrere Seeds klar schlechter ab als der Standardwert (ρ=0,5) - großes ρ zeigt dagegen keinen klaren Nachteil bis ρ=0,95. | `test_rho_experiment_headline_claims` |
| Wie stark hängt die Tourqualität von der Ameisenzahl ab? | Klar: bei sehr wenigen Ameisen (4) bleibt ein Abstand zum Optimum, ab etwa 10 Ameisen wird die kleine Vergleichsinstanz praktisch immer exakt gelöst. | `test_ants_sweep_headline_claims` |
| Stimmen Übergangswahrscheinlichkeit und Pheromon-Update mit der Literatur überein? | Per Handrechnung geprüft; eigene Implementierung UND `acopy` finden auf kleinen Instanzen verlässlich (nahe) das Optimum. | `test_run_aco_and_acopy_reach_a_comparably_good_tour` |

## Ehrliche Grenzen

- **Keine Pheromon-Obergrenze** - ein früh gefundener, mittelmäßiger Pfad kann sich unbegrenzt verstärken, die ganze
  Kolonie legt sich vorzeitig darauf fest (Stagnation), ohne Mechanismus, das zu erkennen oder zu korrigieren. Genau
  das behebt der geplante Fix-Nachfolger Max-Min Ant System (τ_min/τ_max-Schranken) - separates, noch nicht gebautes
  Stück.
- **Der erwartete Nachteil einer zu großen Verdunstungsrate zeigt sich hier nur schwach** - bis ρ=0,95 kein klarer
  Qualitätsverlust gegenüber dem Standardwert gemessen, ehrlich so berichtet statt eine symmetrische U-Form zu
  behaupten.
- **Einfache Pheromon-Initialisierung** ($\tau_0=1$ konstant) statt der literaturüblichen NN-Tour-basierten
  Initialisierung ($\tau_0 = 1/(n \cdot L_{NN})$).
- **Keine Beschleunigungstechniken** (Kandidatenlisten etc.) - Rechenaufwand wächst mit
  $O(\text{Ameisen} \times \text{Stopps}^2)$ je Generation.
- **Letzter unabhängiger Kontrast-Ast dieser Linie** - kein Nachfolger außer dem geplanten MMAS-Fix.

## Tests

55 Tests (`pytest tests/ -v`): Übergangswahrscheinlichkeit/Pheromon-Update per Handrechnung geprüft, Brute-Force-
Vergleich auf sehr kleinen Instanzen (eigene Implementierung UND `acopy` im Vergleich), Szenario-Erzeugung
bitidentisch zu genetic-algorithm-demo/nsga2-demo geprüft, AppTest-Rauchtests (jedes Preset, Generation-Slider inkl.
Abspielen, Permalink-Grenzen, beide Experimente + Sweep auf Abruf) und `test_claims.py` (jede Zahl aus diesem
README, mit CI-robusten Bändern für Einzellauf-Kennzahlen - siehe `feedback_ci_platform_robust_tests.md`, von
Anfang an angewendet).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Einstiegspunkt |
| `aco_constants.py` | Regler-Grenzen, Vehikel-Konstanten, Presets |
| `aco_presets.py` | Permalink/Presets-Mechanik |
| `aco_scenario.py` | Vehikel-Erzeuger (Lieferroute), wortgleich zu genetic-algorithm-demo/nsga2-demo |
| `aco_algorithm.py` | Ant-System-Kern (Übergangswahrscheinlichkeit, Tourkonstruktion, Pheromon-Update) |
| `aco_evaluation.py` | Kennzahlen, Brute-Force-Referenz, Kopfexperiment, Verdunstungsrate-Experiment, Sweep |
| `aco_visualization.py` | Plotly-Abbildungen (Karte mit pheromonstärke-gewichteten Kanten, Vergleiche) |

## Bewusst nicht umgesetzt

- Max-Min Ant System (τ_min/τ_max-Schranken gegen Stagnation) - geplanter Fix-Nachfolger, separates Stück.
- NN-Tour-basierte Pheromon-Initialisierung.
- Kandidatenlisten oder andere Beschleunigungstechniken für größere Instanzen.
- Ein PDF-Export - wie bei den anderen Konzepte-Demos dieses Portfolios nicht Teil der Linie.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Gebaut mit Streamlit, Plotly und numpy.
