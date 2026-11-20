"""Replay the PGCB audit with irrevocable, observation-time decisions.

Both audit policies use identical reported demand, thresholds, feature construction
(notebook 03), frozen Ridge penalty, chronological fits and interval protocol.
The online policy may use the current observation to classify that observation.
Its reference distribution, neighbor and episode evidence contain no later records.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAW_SHA256 = "05c2350f5f3eb398d0b5ca9e9fcdcad31509fa6e1340f8fcef15ec64cbee4dc3"
TEST_START = pd.Timestamp("2024-01-01")
CAL_START = pd.Timestamp("2023-07-01")
HOURS = list(range(1, 24))


def rolling_reference(group, min_count, online, mad=False):
    """Use 29 same-time observations. Only the online reference excludes t."""
    values = group.to_numpy(dtype=float)
    padding = (29, 0) if online else (14, 14)
    padded = np.pad(values, padding, constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, 29)[: len(values)]
    result = np.full(len(group), np.nan)
    sufficient = np.isfinite(windows).sum(axis=1) >= min_count
    selected = windows[sufficient]
    if len(selected):
        medians = np.nanmedian(selected, axis=1)
        if mad:
            medians = np.nanmedian(np.abs(selected - medians[:, None]), axis=1)
        result[sufficient] = medians
    return result


def relative_error(observed, reference):
    return (observed - reference).abs() / reference.abs().clip(lower=1)


def audit(raw, online):
    """Reproduce notebook 01/02 target decisions, with optional past-only references."""
    # The stage-one tests for a regular-hour row are unaffected by 19:30/other
    # rows because their reference is grouped by identical time of day.
    regular = raw.loc[raw.index == raw.index.floor("h")]
    demand = regular.demand_mw
    generated = regular.generation_mw
    supply = generated + regular.load_shedding
    mismatch = (demand - supply).abs()
    severe = (mismatch > 1000) & (
        mismatch / pd.concat([demand.abs(), supply.abs()], axis=1).max(axis=1).clip(lower=1) > 0.15
    )
    clock_hour = demand.index.hour
    demand_groups = demand.groupby(clock_hour)
    local_demand = demand_groups.transform(rolling_reference, 7, online)
    local_generation = generated.groupby(clock_hour).transform(rolling_reference, 7, online)
    demand_deviation = relative_error(demand, local_demand)
    generation_deviation = relative_error(generated, local_generation)
    field_corrupt = demand.isna() | (demand <= 0) | (
        severe & (demand_deviation > 0.60) & (demand_deviation > 2 * generation_deviation)
    )

    reference = demand_groups.transform(rolling_reference, 9, online)
    mad = demand_groups.transform(rolling_reference, 9, online, True)
    difference = (demand - reference).abs()
    deviation = relative_error(demand, reference)
    extreme = (deviation > 0.50) & (difference > 1500)
    # Gaps break episodes; all regular-hour stamps are unique exact integer hours.
    elapsed = demand.index.to_series().diff()
    contiguous = elapsed == pd.Timedelta(hours=1)
    new_group = ~extreme | ~extreme.shift(fill_value=False) | ~contiguous
    groups = extreme.groupby(new_group.cumsum())
    # Online episode size is the observed run length through t.
    length = groups.cumsum() if online else groups.transform("sum")
    previous = demand.shift().where(contiguous)
    if online:
        neighbor = previous
    else:
        subsequent = demand.shift(-1).where(contiguous.shift(-1, fill_value=False))
        neighbor = pd.concat([previous, subsequent], axis=1).median(axis=1)
    isolated = extreme & (relative_error(demand, neighbor) > 0.50) & (
        (demand - neighbor).abs() > 1500
    ) & (length <= 2)

    best = pd.concat(
        [relative_error(demand * scale, reference) for scale in [1.0, 0.01, 0.1, 10.0, 100.0]], axis=1
    ).min(axis=1)
    digit = (demand > 0) & (reference > 0) & (deviation > 0.5) & (
        best <= 0.2
    ) & (deviation / best.clip(lower=1e-9) >= 3)
    lag_day = demand.shift(freq="24h").reindex(demand.index)
    lag_week = demand.shift(freq="168h").reindex(demand.index)
    history = (relative_error(demand, lag_day) > 0.5) & (relative_error(demand, lag_week) > 0.5)
    robust_extreme = (0.6745 * difference / mad.replace(0, np.nan)) > 10
    large = demand > 30000
    tiny = (demand < 1000) & (reference > 3000) & (length <= 2)
    score = (5 * field_corrupt + 3 * digit + 2 * isolated + 1 * history +
             1 * robust_extreme + 2 * severe + 2 * large + 2 * tiny)
    preserve = extreme & (length >= 3) & ~field_corrupt & ~digit & ~severe
    review = (score >= 2) & (score < 4) & ~preserve
    exclude = ((score >= 4) & ~preserve) | field_corrupt | (digit & (isolated | large))
    status = np.select([exclude, review, preserve],
                       ["EXCLUDE_TARGET", "REVIEW", "KEEP_EXTREME_EVENT"], default="KEEP_NORMAL")
    return pd.DataFrame({"demand": demand, "status": status, "usable": demand.notna() & ~exclude & ~review})


def daily_profile(series):
    frame = series.rename("demand").to_frame()
    frame["hour"] = frame.index.hour
    frame["date"] = frame.index.normalize()
    profile = frame[frame.hour.isin(HOURS)].pivot(index="date", columns="hour", values="demand")
    return profile.reindex(index=pd.date_range(profile.index.min(), profile.index.max()),
                           columns=HOURS)


def notebook_features(profile, notebook):
    """Use the unmodified feature-construction section of notebook 03 for each audit."""
    cells = json.loads(notebook.read_text())["cells"]
    source = "\n\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")
    block = source.split("# 15. BUILD TARGET MATRIX", 1)[1].split("# 27. EXTREME-EVENT DAY LABEL", 1)[0]
    context = {"profile": profile, "HOURS": HOURS, "np": np, "pd": pd, "print": lambda *args: None}
    exec(compile(block, str(notebook), "exec"), context)
    groups = ["calendar", "lag1", "lag7", "roll7", "roll28", "roll7_std", "roll28_std", "summary_features"]
    features = pd.concat([context[name] for name in groups], axis=1).astype(float)
    if features.shape[1] != 165:
        raise ValueError("Feature schema no longer matches the frozen 165-feature protocol.")
    return features


def ridge_predictions(features, labels, fitting, prediction_dates):
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit(features.loc[fitting], labels.loc[fitting])
    return model.predict(features.loc[prediction_dates])


def interval_metrics(features, labels, eligible, forecast_dates, use_regime):
    """Predict every feature-ready test day; append a residual only after a complete label day."""
    proper = eligible & (features.index < CAL_START)
    calibration = features.index[eligible & (features.index >= CAL_START) & (features.index < TEST_START)]
    dates = calibration.append(forecast_dates)
    predict = ridge_predictions(features, labels, proper, dates)
    scale = np.maximum(np.abs(predict), 1000.0)
    y = labels.loc[dates].to_numpy()
    errors = np.abs(y - predict) / scale
    trend = features.lag1_daily_mean / features.filter(regex="^roll28mean_h").mean(axis=1).clip(lower=1)
    lower_cut, upper_cut = trend.loc[proper].quantile([1/3, 2/3])
    regimes = np.select([trend.loc[dates] < lower_cut, trend.loc[dates] > upper_cut], [-1, 1], default=0)
    observed = list(range(len(calibration)))
    lower, upper = [], []
    for row in range(len(calibration), len(dates)):
        history = np.asarray(observed[-180:])
        same_regime = history[regimes[history] == regimes[row]]
        if use_regime and len(same_regime) >= 25:
            history = same_regime
        sample = errors[history]
        rank = min(len(history), int(np.ceil(0.9 * (len(history) + 1)))) - 1
        q = np.partition(sample, rank, axis=0)[rank]
        lower.append(np.maximum(predict[row] - q * scale[row], 0))
        upper.append(predict[row] + q * scale[row])
        if np.isfinite(y[row]).all():
            observed.append(row)
    return np.array(lower), np.array(upper)


def forecast(audited, raw_profile, notebook):
    labels = daily_profile(audited.demand.where(audited.usable))
    features = notebook_features(labels, notebook)
    history_ok = features.notna().all(axis=1)
    eligible = history_ok & labels.notna().all(axis=1)
    before_test = features.index < TEST_START
    dates = features.index[~before_test & history_ok]
    ridge = ridge_predictions(features, labels, before_test & eligible, dates)
    d1 = features.loc[dates].filter(regex="^lag1_h").to_numpy()
    observed = raw_profile.loc[dates].to_numpy()
    err_ridge = ridge - observed
    err_d1 = d1 - observed
    frame = pd.DataFrame({
        "own_eligible": eligible.loc[dates],
        "raw_target_complete": np.isfinite(observed).all(axis=1),
        "ridge_mae": np.abs(err_ridge).mean(axis=1), "d1_mae": np.abs(err_d1).mean(axis=1),
        "ridge_mse": (err_ridge ** 2).mean(axis=1),
    }, index=dates)
    for method, conditioning in [("hour", False), ("regime_hour", True)]:
        lo, hi = interval_metrics(features, labels, eligible, dates, conditioning)
        frame[f"{method}_coverage"] = np.where(
            frame.raw_target_complete, ((lo <= observed) & (observed <= hi)).mean(axis=1), np.nan
        )
        frame[f"{method}_width"] = (hi - lo).mean(axis=1)
    splits = pd.Series(np.select([features.index < "2023", before_test], ["train", "validation"], default="test"),
                       index=features.index)
    stats = pd.DataFrame({"eligible_days": eligible.groupby(splits).sum(),
                          "feature_ready_days": history_ok.groupby(splits).sum(),
                          "complete_target_days": labels.notna().all(axis=1).groupby(splits).sum()})
    return frame.rename_axis("target_date"), stats


def summarize(daily, mask):
    sample = daily.loc[mask]
    mae, d1 = sample[["ridge_mae", "d1_mae"]].mean()
    return dict(days=len(sample), ridge_mae_mw=mae, d1_mae_mw=d1, skill_vs_d1_pct=100*(1-mae/d1),
                ridge_rmse_mw=np.sqrt(sample.ridge_mse.mean()),
                coverage_hour_pct=100*sample.hour_coverage.mean(),
                coverage_regime_hour_pct=100*sample.regime_hour_coverage.mean(),
                width_hour_mw=sample.hour_width.mean(), width_regime_hour_mw=sample.regime_hour_width.mean())


def paired_bootstrap(retrospective, online, dates):
    """Percentile intervals for loss changes; blocks contain consecutive eligible dates."""
    variables = ["ridge_mae", "hour_coverage", "regime_hour_coverage"]
    differences = (online.loc[dates, variables] - retrospective.loc[dates, variables]).to_numpy()
    differences[:, 1:] *= 100
    units = ["MW", "percentage points", "percentage points"]
    rng = np.random.default_rng(42)
    results = []
    for block in (7, 14):
        length = len(dates)
        replicate = np.empty((5000, len(variables)))
        blocks_needed = int(np.ceil(length / block))
        for index in range(len(replicate)):
            starts = rng.integers(0, length - block + 1, size=blocks_needed)
            selected = (starts[:, None] + np.arange(block)).ravel()[:length]
            replicate[index] = differences[selected].mean(axis=0)
        for column, variable in enumerate(variables):
            low, high = np.quantile(replicate[:, column], [0.025, 0.975])
            results.append({"metric": variable, "units": units[column], "block_eligible_days": block,
                            "paired_dates": length, "mean_past_only_minus_retrospective": differences[:, column].mean(),
                            "lower_95": low, "upper_95": high})
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--notebooks", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--reference-audit", type=Path, help="Optional original notebook 02 regular-hour audit CSV")
    args = parser.parse_args()
    if hashlib.sha256(args.raw.read_bytes()).hexdigest() != RAW_SHA256:
        raise ValueError("Raw CSV differs from the released PGCB V1 input.")
    raw = pd.read_csv(args.raw, parse_dates=["datetime"]).set_index("datetime").sort_index()
    if not raw.index.is_unique:
        raise ValueError("PGCB V1 timestamps must be unique.")
    policies = {"retrospective": audit(raw, False), "past_only": audit(raw, True)}
    if args.reference_audit:
        reference = pd.read_csv(args.reference_audit, usecols=["datetime", "advanced_quality_status"],
                                parse_dates=["datetime"]).set_index("datetime")
        pd.testing.assert_series_equal(policies["retrospective"].status,
                                       reference.advanced_quality_status, check_names=False)
    # Cropping at several forecast origins cannot change any past-only status or label.
    for origin in ["2023-12-31 23:00", "2024-05-20 16:00", "2025-08-01 23:00"]:
        pd.testing.assert_frame_equal(audit(raw.loc[:origin], True), policies["past_only"].loc[:origin])
    raw_profile = daily_profile(raw.demand_mw[raw.index == raw.index.floor("h")])
    args.out.mkdir(parents=True, exist_ok=True)
    split_tables, predictions = [], {}
    notebook = args.notebooks / "03_Forecasting_Dataset_and_EDA.ipynb"
    for name, audited in policies.items():
        predictions[name], split = forecast(audited, raw_profile, notebook)
        split.insert(0, "audit", name)
        split_tables.append(split.rename_axis("split"))
    pd.concat(split_tables).reset_index().to_csv(args.out / "audit_splits.csv", index=False)
    status = pd.concat({k: v.status for k,v in policies.items()}, axis=1)
    status.groupby(["retrospective","past_only"]).size().rename("regular_hour_rows").to_csv(args.out / "audit_crosstab.csv")
    changed_labels = policies["retrospective"].usable != policies["past_only"].usable
    status.loc[changed_labels].to_csv(args.out / "audit_changed_target_status.csv")
    outcomes = []
    common = predictions["retrospective"].query("own_eligible").index.intersection(
        predictions["past_only"].query("own_eligible").index)
    common_raw = predictions["retrospective"].query("raw_target_complete").index.intersection(
        predictions["past_only"].query("raw_target_complete").index)
    for name, daily in predictions.items():
        scopes = {"own_eligible": daily.own_eligible, "common_eligible": daily.index.isin(common),
                  "all_raw_complete_common_history": daily.index.isin(common_raw)}
        for scope, mask in scopes.items():
            outcomes.append({"audit": name, "population": scope, **summarize(daily, mask)})
    results = pd.DataFrame(outcomes)
    original = results.iloc[0]
    if (original.days != 709 or round(original.ridge_mae_mw, 2) != 437.16 or
        round(original.coverage_regime_hour_pct, 2) != 90.35 or round(original.coverage_hour_pct, 2) != 89.20):
        raise ValueError(f"Original benchmark reproduction failed: {original.to_dict()}")
    results.to_csv(args.out / "audit_metrics.csv", index=False, float_format="%.8f")
    pd.concat(predictions, names=["audit","target_date"]).to_csv(args.out / "audit_daily.csv", float_format="%.8f")
    changes = paired_bootstrap(predictions["retrospective"], predictions["past_only"], common)
    changes.to_csv(args.out / "audit_paired_changes.csv", index=False, float_format="%.8f")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
