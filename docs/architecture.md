# Architecture: where this toolkit comes from

This toolkit is the shared infrastructure layer of a private multi-sport
prediction platform covering six sports (baseball, basketball, football,
hockey, soccer, tennis). The platform runs a daily cycle: ingest results,
recompute ratings and features, score upcoming games, calibrate, size
positions, and execute against prediction markets — with web dashboards
monitoring every stage.

The platform separates cleanly into two layers, and that boundary is
exactly what's public here:

```
                       ┌─────────────────────────────────────────────┐
                       │                PRIVATE                      │
                       │                                             │
  game data ──────────►│  feature engineering        per-sport       │
                       │  (the edge: which signals,  models +        │
                       │  how they're computed)      configs         │
                       └───────────────┬─────────────────────────────┘
                                       │ probabilities
                       ┌───────────────▼─────────────────────────────┐
                       │           PUBLIC (this repo)                │
                       │                                             │
   pmt.ratings ───────►│  base models ──► stacking ──► calibration   │
   (Elo inputs)        │                 [pmt.ensembling] [pmt.calibration]
                       │                                  │          │
                       │  evaluation everywhere           ▼          │
                       │  [pmt.evaluation]      trust weights ──► stakes
                       │                              [pmt.staking]  │
                       │  orchestrated by [pmt.pipeline]             │
                       └─────────────────────────────────────────────┘
                                       │ sized positions
                                       ▼
                          prediction-market execution
```

The litmus test for what's public: *could a reader infer the specific
feature engineering or model design from this code?* The mathematics of
calibration, the mechanics of stacking, Kelly staking, and Elo are
textbook material — the value demonstrated here is implementation
quality. Which features feed the models, how they're computed, and how
the production ensembles are composed stay private.

## Stage by stage

**Ratings (`pmt.ratings`).** Elo is a foundational input feature in every
sport. Production needs more than the textbook update rule: K factors
that decay as a season matures, regression toward the mean at season
boundaries, margin-of-victory adjustments that don't inflate favorites,
and per-team uncertainty (rating deviation) so that early-season results
against unknown opponents count for less.

**Ensembling (`pmt.ensembling`).** Per-sport base models are combined by
a meta-learner. With time-ordered data the cardinal failure mode is
temporal leakage — a base model that has seen the meta-model's training
rows produces optimistically-biased predictions there, and the ensemble's
offline metrics inflate silently. The framework makes the safe pattern
structural: a four-window chronological split and a guard that rejects
base models trained beyond their window *before* anything runs.

**Calibration (`pmt.calibration`).** A model can rank outcomes well and
still be systematically over- or under-confident — fatal when the
output is compared against market prices, because miscalibration reads
as phantom edge. Every production probability passes through a
calibration curve fitted on held-out data (isotonic / Platt / temperature
/ beta, selected per model by held-out metrics). Curves serialize as
plain dicts so the fitting environment and the serving environment share
nothing but JSON.

**Evaluation (`pmt.evaluation`).** Brier score, log-loss, and expected
calibration error, computed under splits that respect time. The
train/calibrate/evaluate helper exists so that "did calibration help?"
is always answered on a window neither the model nor the calibrator saw.

**Staking (`pmt.staking`).** Probabilities become positions through a
modified Kelly criterion with three safety systems learned the hard way:
skill-aware shrinkage toward the market price (a model that measures
better than the market is still treated only as its equal), an edge gate
that zeroes out noise-level edges, and empirical bin trust weights that
scale stakes by whether the edge has *actually materialized* as realized
P&L in each probability range — replacing hand-tuned heuristics like a
fixed longshot penalty with measured evidence.

**Pipeline (`pmt.pipeline`).** The daily cycle is a sequence of steps
with skip conditions (dry runs), per-step failure policy (a third-party
enrichment timing out shouldn't kill the run), and parallel fan-out for
independent computations. Deliberately smaller than a DAG scheduler;
deliberately bigger than a shell script.

## Design rules

1. **Arrays and dicts in, arrays and dicts out.** No module imports a
   database driver. Storage, model registries, and data access belong to
   the caller — which is why these modules extracted cleanly in the
   first place.
2. **Serializable artifacts.** Calibration curves, trust weights, and
   stake recommendations are plain dicts. Anything that needs to cross a
   process boundary survives JSON.
3. **Factories over instances.** Anything that evaluates models takes a
   zero-arg factory, so the toolkit never knows how callers construct,
   configure, or version their models.
4. **Temporal discipline is structural, not conventional.** Splits that
   respect time are enforced by code (degenerate-split errors, the
   anti-leakage guard), not by documentation asking nicely.

## Dashboards

The private platform's web app (Flask) monitors each stage: calibration
curves and bin health per league, ensemble configuration, market P&L by
probability bin, and rating management. Screenshots in the
[gallery](gallery/README.md) show these running against live data with
internal identifiers removed; every chart in the gallery generated by
this repo's examples is synthetic and reproducible.
