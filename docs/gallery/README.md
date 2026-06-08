# Gallery

## Toolkit-generated charts

All charts below are produced entirely from **synthetic data** by scripts in
`examples/` — regenerate with:

```bash
python examples/01_calibration_methods.py
python examples/make_gallery.py
```

| Image | What it shows |
|---|---|
| `calibration_reliability_diagram.png` | Raw overconfident predictions vs. all four calibration methods on a held-out window — the classic reliability diagram. |
| `calibration_fitted_curve_shapes.png` | The fitted probability mappings themselves: isotonic's step function vs. the smooth sigmoid/temperature/beta families. |
| `calibration_empirical_bins_vs_isotonic.png` | Empirical 5%-wide calibration bins (marker area ∝ sample count) with the isotonic fit through them. |
| `calibration_temperature_scaling_family.png` | The temperature-scaling family — how T>1 softens and T<1 sharpens probabilities, with the fitted T highlighted. |
| `evaluation_temporal_split_raw_vs_calibrated.png` | Brier / log-loss / ECE before and after calibration under a leakage-safe temporal train/calibrate/evaluate split. |
| `staking_bin_trust_kelly_pipeline.png` | The probability → action pipeline: bin trust weights (raw ROI → shrunk → smoothed) and a sized slate showing trust damping longshot stakes. |
| `ratings_elo_convergence_carryover.png` | Two simulated seasons of Elo: dynamic-K trajectories with the carryover regression visible at the season boundary, and final ratings recovering latent strength (r ≈ 0.97). |
| `ensembling_stacking_vs_base_models.png` | Temporal stacking under a 3000/1500/750/750 split: the ensemble beats every base model on the untouched evaluation window. |

## ParaView renders

3D scientific visualizations rendered in [ParaView](https://www.paraview.org/).
The underlying data is generated from synthetic inputs by a script in
`examples/paraview/` (so it stays reproducible and proprietary-free); the
render itself is produced interactively in the ParaView GUI, then exported
here.

| Image | What it shows | Data script |
|---|---|---|
| `paraview_probability_surface.png` | A 2-feature model's **calibrated win-probability surface** over its full input space. Height and color both map to `P(outcome)` (blue ≈ 0 → white = 0.5 decision boundary → red ≈ 1). The saddle curvature is an interaction term between the two features; the white ridge is the model's coin-flip frontier. Pipeline: CSV → Table To Points → Delaunay 2D → Warp By Scalar → Contour. | `examples/paraview/make_probability_surface_csv.py` |

## Dashboard screenshots

Screenshots of the private multi-sport prediction platform's web dashboards
(calibration monitoring, market analytics, rating management) will be added
here. They are captured from the live system with any internal identifiers
removed.
