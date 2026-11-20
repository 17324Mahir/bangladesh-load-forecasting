# Reproduce the ICCIT reviewer checks

Run these commands in order from the repository root on `main`, using the same terminal for all three steps. The original notebooks come from [commit `65c5199`](https://github.com/17324Mahir/bangladesh-load-forecasting/commit/65c5199aae72f71f87b1a24ecfaff4ac7a0fb21d). The scripts reuse their preprocessing, features, conformal code and Keras model definitions. The local preparation skips Colab mounting, interactive upload and plotting.

## 1. Rebuild the original inputs

The input is `pgcb_V1_raw.csv` from Shekh and Rafi’s [PGCB V1 archive](https://doi.org/10.17632/vpk8spw2mm.1), SHA-256 `05c2350f5f3eb398d0b5ca9e9fcdcad31509fa6e1340f8fcef15ec64cbee4dc3`. The download requires curl. You can also put the raw CSV at `$RUN/data/raw/pgcb_V1_raw.csv` yourself. Use an empty work directory for a fresh run.

```sh
RUN="$PWD/../iccit-reproduction"
python3.13 -m venv "$RUN/analysis-env"
A="$RUN/analysis-env/bin/python"
"$A" -m pip install -r reviewer_checks/requirements-audit.txt
"$A" reviewer_checks/fetch_pgcb.py --project "$RUN"
"$A" reviewer_checks/original_notebooks.py --notebooks notebooks --project "$RUN"
```

Notebook 01 reads the raw PGCB release, 02 finalizes the hourly target, and 03 builds the 23-slot profiles and forecasting splits. The expected original split is 2,236 training, 273 validation and 709 test days.

## 2. Reproduce the audit and uncertainty results

```sh
"$A" reviewer_checks/audit_counts.py --project "$RUN" --out "$RUN/results/audit_counts"
"$A" reviewer_checks/audit_replay.py --raw "$RUN/data/raw/pgcb_V1_raw.csv" --notebooks notebooks \
  --reference-audit "$RUN/data/processed/pgcb_regular_hours_advanced_audit.csv" --out "$RUN/results/causal"
"$A" reviewer_checks/conformal_sensitivity.py --project "$RUN" --notebooks notebooks --out "$RUN/results/conformal"
```

- [Audit counts](results/audit_counts/counts.csv) and [cross-tabulation](results/audit_counts/decisions_by_stage.csv) join decisions by original row ID. The first-stage severity flag and final target status can both apply to the same record.
- [Audit replay](results/causal/audit_metrics.csv) rebuilds the original and past-only features, refits the Ridge models, and uses the same raw observations for scoring on common dates. The optional reference check compares every original hourly classification. The past-only policy makes each quality decision when a record arrives. The [paired differences](results/causal/audit_paired_changes.csv) use 5,000 moving-block resamples of daily losses in eligible-date order, with 7- and 14-day blocks.
- [Point-model reproduction](results/conformal/point_model_reproduction.csv) distinguishes the full point-model refit from the nested Ridge used for calibration. [Interval sensitivity](results/conformal/conformal_sensitivity.csv) varies one setting at a time. Windows count eligible days, starting with up to 143 calibration days. The 90% target stays fixed. A regime with fewer than 25 historical days uses pooled hour-specific quantiles. The [regime results](results/conformal/conformal_by_regime.csv) record how often that fallback occurred.

## 3. Reproduce the matched-input comparator

Use the same `RUN` directory. The analysis code used Python 3.13.7. The Keras refits used Python 3.12.14 on CPU with TensorFlow 2.20.0. The two dependency files record the numerical environments used for these runs.

```sh
python3.12 -m venv "$RUN/neural-env"
N="$RUN/neural-env/bin/python"
"$N" -m pip install -r reviewer_checks/requirements-matched.txt
"$N" reviewer_checks/matched_models.py --forecast-dir "$RUN/data/forecasting" \
  --notebook notebooks/05_Deep_Sequence_Models.ipynb --output "$RUN/results/matched" --threads 2
"$N" reviewer_checks/matched_models.py --output "$RUN/results/matched" --summary-only
```

All five models use identical 28-day histories (23 loads and 23 observation masks per day), eight calendar values, 23 outputs and the same dates. Ridge flattens the history and selects its penalty on validation. Each neural model is refitted on training plus validation at seeds 42–46 and its original validation-selected epoch count (LSTM 32, GRU 12, temporal CNN 76, Transformer 32). The matched split has 2,234 training, 273 validation and 709 test dates.

The [per-seed results](results/matched/matched_per_seed.csv), [model summary](results/matched/matched_summary.csv) and [Ridge selection grid](results/matched/ridge_validation_grid.csv) are the outputs used in the revision. The summary uses the arithmetic mean and sample standard deviation across the five runs. The public tables contain derived results. The local run also saves per-run predictions, targets, dates and settings for verification. Floating-point results can vary with platform and library versions.
