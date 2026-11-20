"""Matched-information Ridge and seed sensitivity for the notebook 05 architectures."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import sklearn


EPOCHS = {"LSTM": 32, "GRU": 12, "TemporalCNN": 76, "Transformer": 32}
PARAMETERS = {"LSTM": 125175, "GRU": 103159, "TemporalCNN": 100247, "Transformer": 99383}
SEEDS = (42, 43, 44, 45, 46)
ALPHAS = (0.01, 0.1, 1, 10, 100, 1000)


def make_inputs(directory):
    """Construct the same 28-day histories, eight calendar features and 23 targets as 05."""
    profiles = pd.read_csv(directory / "daily_23slot_load_profile_raw.csv")
    date_name = "target_date" if "target_date" in profiles else "date"
    profiles = profiles.set_index(pd.to_datetime(profiles[date_name])).sort_index()
    days = pd.date_range(profiles.index.min(), profiles.index.max())
    hours = range(1, 24)
    raw = profiles.reindex(days)[[f"demand_h{h:02}" for h in hours]].to_numpy(float)
    observed = np.isfinite(raw)
    filled = raw.copy()
    for i, h in zip(*np.where(~observed)):
        if i >= 7 and np.isfinite(filled[i - 7, h]):
            filled[i, h] = filled[i - 7, h]
        elif i >= 1 and np.isfinite(filled[i - 1, h]):
            filled[i, h] = filled[i - 1, h]
        else:
            prior = filled[max(0, i - 28):i, h]
            if np.isfinite(prior).any():
                filled[i, h] = np.nanmedian(prior)

    samples = pd.read_csv(directory / "forecasting_samples_23slot.csv", parse_dates=["target_date"])
    samples = samples.sort_values("target_date")
    if samples.target_date.duplicated().any():
        raise ValueError("Duplicate dates in forecast samples.")
    ix = days.get_indexer(samples.target_date)
    if (ix < 0).any():
        raise ValueError("Forecast samples contain dates absent from the profile calendar.")
    samples = samples.loc[ix >= 28].copy()
    ix = ix[ix >= 28]
    history_rows = ix[:, None] + np.arange(-28, 0)
    y = samples[[f"target_h{h:02}" for h in hours]].to_numpy(np.float32)
    keep = np.isfinite(filled[history_rows]).all(axis=(1, 2)) & np.isfinite(y).all(axis=1)
    history_rows, y = history_rows[keep], y[keep]
    samples = samples.loc[keep]
    dates = pd.DatetimeIndex(samples.target_date)
    if not np.array_equal(y, raw[ix[keep]].astype(np.float32)):
        raise ValueError("Raw daily labels and forecasting samples disagree.")
    week = 2 * np.pi * dates.dayofweek / 7
    month = 2 * np.pi * dates.month / 12
    year = 2 * np.pi * dates.dayofyear / 365.25
    calendar = np.column_stack((np.sin(week), np.cos(week), np.sin(month), np.cos(month),
                                np.sin(year), np.cos(year), dates.dayofweek.isin([4, 5]),
                                (dates - pd.Timestamp("2015-01-01")).days / 365.25 / 10)).astype(np.float32)
    splits = samples.split.to_numpy()
    counts = {split: int(np.sum(splits == split)) for split in ("TRAIN", "VALIDATION", "TEST")}
    if list(counts.values()) != [2234, 273, 709]:
        raise ValueError(f"Dates do not reproduce the published sequence splits: {counts}")
    return filled[history_rows].astype(np.float32), observed[history_rows].astype(np.float32), calendar, y, dates, splits


def normalize(loads, masks, y, fit_mask):
    mean = y[fit_mask].mean(axis=0)
    std = y[fit_mask].std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    sequence = np.concatenate(((loads - mean) / std, masks), axis=-1)
    return sequence, (y - mean) / std, mean, std


def define_models(notebook, tf):
    """Reuse the original Keras function definitions, without Drive or notebook side effects."""
    names = {"positional_encoding", "output_head", "build_lstm_model", "build_gru_model",
             "build_cnn_model", "transformer_block", "build_transformer_model", "compile_model"}
    cells = json.loads(notebook.read_text())["cells"]
    functions = []
    for cell in cells:
        if cell["cell_type"] == "code":
            tree = ast.parse("".join(cell["source"]))
            functions.extend(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names)
    if {node.name for node in functions} != names:
        raise ValueError("Notebook 05 does not contain the expected Keras builders.")
    namespace = dict(tf=tf, keras=tf.keras, layers=tf.keras.layers, np=np, SEQUENCE_LENGTH=28)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(notebook), "exec"), namespace)
    return namespace


def metrics(actual, prediction):
    error = prediction.astype(np.float64) - actual
    return dict(mae_mw=np.mean(np.abs(error)), rmse_mw=np.sqrt(np.mean(error ** 2)),
                mape_pct=np.mean(np.abs(error / actual)) * 100)


def save_run(output, name, seed, dates, actual, prediction, extra):
    """The JSON is the completion marker, written after the matching predictions."""
    key = f"{name}_{seed}" if seed is not None else name
    if not np.isfinite(prediction).all() or actual.shape != prediction.shape:
        raise ValueError(f"Invalid predictions for {key}")
    forecasts = output / f"{key}.npz"
    temp = output / f".{key}.npz"
    np.savez_compressed(temp, date=dates.to_numpy(dtype="datetime64[D]"),
                        actual_mw=actual, prediction_mw=prediction)
    temp.replace(forecasts)
    row = dict(model=name, seed=seed, n_train=2234, n_validation=273, n_test=len(actual),
               **metrics(actual, prediction), **extra)
    record = output / f"{key}.json"
    temporary = record.with_suffix(".tmp")
    temporary.write_text(json.dumps(row, indent=2) + "\n")
    temporary.replace(record)
    print(f"{key}: test MAE {row['mae_mw']:.3f} MW", flush=True)


def summarize(output):
    records = []
    with np.load(output / "Ridge.npz") as reference:
        dates = reference["date"]
        actual = reference["actual_mw"]
    for path in sorted(output.glob("*.json")):
        if path.stem != "Ridge" and path.stem.split("_")[0] not in EPOCHS:
            continue
        row = json.loads(path.read_text())
        with np.load(path.with_suffix(".npz")) as run:
            np.testing.assert_array_equal(run["date"], dates)
            np.testing.assert_array_equal(run["actual_mw"], actual)
            for metric, value in metrics(actual, run["prediction_mw"]).items():
                np.testing.assert_allclose(value, row[metric], rtol=1e-10)
        records.append(row)
    results = pd.DataFrame(records).sort_values(["model", "seed"])
    keysets = results.groupby("model").seed.agg(lambda values: set(values.dropna().astype(int)))
    if "Ridge" not in keysets or any(keysets.get(model) != set(SEEDS) for model in EPOCHS):
        raise ValueError("Summary requires the Ridge run and the full prespecified five seeds for each architecture.")
    results.to_csv(output / "matched_per_seed.csv", index=False)
    summary = results.groupby("model", as_index=False).agg(
        n_fits=("model", "size"), mae_mw=("mae_mw", "mean"), mae_seed_sd_mw=("mae_mw", "std"),
        rmse_mw=("rmse_mw", "mean"), rmse_seed_sd_mw=("rmse_mw", "std"),
        mape_pct=("mape_pct", "mean"), mape_seed_sd_pct=("mape_pct", "std"))
    summary.to_csv(output / "matched_summary.csv", index=False)
    print(summary.to_string(index=False))


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        summarize(output)
        return
    loads, masks, calendar, y, dates, splits = make_inputs(args.forecast_dir)
    train, val, test = (splits == split for split in ("TRAIN", "VALIDATION", "TEST"))
    trainval = train | val
    sequence, target, mean, std = normalize(loads, masks, y, trainval)
    versions = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    sklearn=sklearn.__version__)
    if "Ridge" in args.models:
        x_train, y_train, train_mean, train_std = normalize(loads, masks, y, train)
        x_train = np.column_stack((x_train.reshape(len(x_train), -1), calendar)).astype(np.float64)
        scores = []
        for alpha in ALPHAS:
            ridge = Ridge(alpha=alpha, solver="cholesky").fit(x_train[train], y_train[train])
            pred = ridge.predict(x_train[val]) * train_std + train_mean
            scores.append(dict(alpha=alpha, validation_mae_mw=metrics(y[val], pred)["mae_mw"]))
        pd.DataFrame(scores).to_csv(output / "ridge_validation_grid.csv", index=False)
        best = min(scores, key=lambda row: row["validation_mae_mw"])
        x_final = np.column_stack((sequence.reshape(len(sequence), -1), calendar)).astype(np.float64)
        ridge = Ridge(alpha=best["alpha"], solver="cholesky").fit(x_final[trainval], target[trainval])
        pred = ridge.predict(x_final[test]) * std + mean
        save_run(output, "Ridge", None, dates[test], y[test], pred, {**best, **versions})

    neural_models = [name for name in args.models if name != "Ridge"]
    if not neural_models:
        return
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    scope = define_models(args.notebook, tf)
    builders = {name: scope[f"build_{key}_model"] for name, key in (
        ("LSTM", "lstm"), ("GRU", "gru"), ("TemporalCNN", "cnn"), ("Transformer", "transformer"))}
    versions.update(tensorflow=tf.__version__, keras=tf.keras.__version__,
                    notebook_sha256=hashlib.sha256(args.notebook.read_bytes()).hexdigest(),
                    intraop_threads=args.threads)
    fit_inputs = [sequence[trainval], calendar[trainval]]
    test_inputs = [sequence[test], calendar[test]]
    for seed in args.seeds:
        for name in neural_models:
            print(f"Starting {name}, seed {seed}, {EPOCHS[name]} validation-selected epochs", flush=True)
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed)
            model = scope["compile_model"](builders[name](sequence_shape=(28, 46), calendar_dim=8))
            if model.count_params() != PARAMETERS[name]:
                raise ValueError(f"Parameter count mismatch for {name}")
            start = time.monotonic()
            model.fit(fit_inputs, target[trainval], epochs=EPOCHS[name],
                      batch_size=32, shuffle=True, verbose=args.verbose)
            prediction = model.predict(test_inputs, batch_size=32, verbose=0) * std + mean
            save_run(output, name, seed, dates[test], y[test], prediction, dict(
                epochs=EPOCHS[name], parameters=model.count_params(), elapsed_seconds=round(time.monotonic()-start,2),
                **versions))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--forecast-dir", type=Path, default=Path("data/forecasting"))
    ap.add_argument("--notebook", type=Path, default=Path("notebooks/05_Deep_Sequence_Models.ipynb"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--models", nargs="+", choices=["Ridge", *EPOCHS], default=["Ridge", *EPOCHS])
    ap.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=SEEDS)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--verbose", type=int, choices=[0, 2], default=2)
    ap.add_argument("--summary-only", action="store_true")
    run(ap.parse_args())
