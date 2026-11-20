"""Review sensitivity for the original hour-specific rolling 90% intervals."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from original_notebooks import run_notebook


def regime_usage(initial, test, window, minimum):
    """Count day-level fallback in each regime using only earlier outcomes."""
    history = list(initial)
    rows = []
    for regime in test:
        previous = history[-window:]
        support = sum(label == regime for label in previous)
        rows.append((regime, support, support < minimum))
        history.append(regime)
    return pd.DataFrame(rows, columns=["regime", "support_days", "fallback"])


def check_point_models(state):
    features, targets = state["FEATURE_COLUMNS"], state["TARGET_COLUMNS"]
    train, val, test = (state[name] for name in ("train_df", "validation_df", "test_df"))
    # Reproduce Notebook 04's Table II ridge, refit on training + validation
    # after selecting its penalty on 2023 validation. Notebook 07's separate
    # model holds July–December 2023 out for interval calibration.
    grid = state["RIDGE_ALPHAS"]
    scaler = StandardScaler().fit(train[features])
    x_train, x_val = scaler.transform(train[features]), scaler.transform(val[features])
    val_mae = {}
    for alpha in grid:
        fit = Ridge(alpha=alpha).fit(x_train, train[targets])
        val_mae[alpha] = np.abs(val[targets].to_numpy() - fit.predict(x_val)).mean()
    best = min(val_mae, key=val_mae.get)
    train_val = pd.concat([train, val], ignore_index=True)
    scaler = StandardScaler().fit(train_val[features])
    model = Ridge(alpha=best).fit(scaler.transform(train_val[features]), train_val[targets])
    actual = test[targets].to_numpy()
    ridge = model.predict(scaler.transform(test[features]))
    naive = test[state["LAG1_FEATURES"]].to_numpy()
    rows = []
    for model_name, predictions, alpha, days in [
        ("D1", naive, None, 0),
        ("Ridge, full 2023 refit", ridge, best, len(train_val)),
        ("Ridge, nested calibration", state["TEST_PRED"], state["BEST_ALPHA"], len(state["proper_train_df"])),
    ]:
        errors = np.abs(actual - predictions)
        rows.append(dict(model=model_name, alpha=alpha, fit_days=days, test_days=len(test),
                         mae_mw=errors.mean(), rmse_mw=np.sqrt(np.mean(errors**2)),
                         mape_pct=100 * np.mean(errors / np.abs(actual))))
    return pd.DataFrame(rows)


def evaluate_variant(state, fold, window, floor, quantiles):
    """Use notebook 07's interval constructor and evaluation functions verbatim."""
    low, high = np.quantile(state["PROPER_REGIME_SIGNAL"], quantiles)
    state.update(LOW_THRESHOLD=low, HIGH_THRESHOLD=high)
    for split in ("CAL", "TEST"):
        predictions = state[f"{split}_PRED"]
        state[f"{split}_REGIME"] = state["assign_regime"](state[f"{split}_REGIME_SIGNAL"])
        scale = np.maximum(np.abs(predictions), floor)
        state[f"{split}_SCALE"] = scale
        state[f"{split}_SCORE"] = state[f"{split}_ABS_ERROR"] / scale
        state[f"{split}_PROFILE_SCORE"] = state[f"{split}_SCORE"].max(axis=1)

    lower, upper = state["build_adaptive_intervals"](0.90, window, regime_aware=True)
    metrics = state["evaluate_interval"](state["Y_test"], lower, upper, 0.90)
    usage = regime_usage(state["CAL_REGIME"], state["TEST_REGIME"], window, state["MIN_REGIME_SAMPLES"])
    result = dict(
        variant=fold, window_days=window, floor_mw=floor, quantile_low=quantiles[0],
        quantile_high=quantiles[1], threshold_low=low, threshold_high=high,
        initial_calibration_days=len(state["CAL_PRED"]), test_days=len(usage),
        min_cal_abs_forecast_mw=np.abs(state["CAL_PRED"]).min(),
        min_test_abs_forecast_mw=np.abs(state["TEST_PRED"]).min(),
        cal_forecasts_below_floor=(np.abs(state["CAL_PRED"]) < floor).sum(),
        test_forecasts_below_floor=(np.abs(state["TEST_PRED"]) < floor).sum(),
        fallback_days=usage["fallback"].sum(),
        coverage_pct=metrics["Marginal_Coverage_percent"], width_mw=metrics["MPIW_MW"],
        winkler_mw=metrics["Winkler_Score"],
    )
    regime_rows = []
    for regime in ("LOW", "NORMAL", "HIGH"):
        mask = usage["regime"].eq(regime).to_numpy()
        local = state["evaluate_interval"](state["Y_test"][mask], lower[mask], upper[mask], 0.90)
        regime_rows.append(dict(variant=fold, regime=regime, test_days=mask.sum(),
                                fallback_days=usage.loc[mask, "fallback"].sum(),
                                coverage_pct=local["Marginal_Coverage_percent"],
                                width_mw=local["MPIW_MW"], winkler_mw=local["Winkler_Score"]))
    return result, regime_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebooks", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    state = run_notebook(args.notebooks, args.project, "07")
    points = check_point_models(state)
    thirds = (1/3, 2/3)
    settings = [
        ("primary", 180, 1000, thirds),
        ("window_90", 90, 1000, thirds),
        ("window_365", 365, 1000, thirds),
        ("floor_500", 180, 500, thirds),
        ("floor_2000", 180, 2000, thirds),
        ("quartile_cutoffs", 180, 1000, (0.25, 0.75)),
    ]
    results, regimes = [], []
    for settings_row in settings:
        metrics, local = evaluate_variant(state, *settings_row)
        results.append(metrics)
        regimes.extend(local)
    summary = pd.DataFrame(results)

    # Anchor local execution to the precision in the submitted manuscript.
    observed = np.r_[points["mae_mw"].to_numpy(), summary.iloc[0][["coverage_pct","width_mw","winkler_mw"]].astype(float)]
    published = [649.20, 437.16, 441.44, 90.35, 2042.42, 2966.04]
    np.testing.assert_allclose(observed, published, rtol=0, atol=0.0051)
    args.out.mkdir(parents=True, exist_ok=True)
    points.to_csv(args.out / "point_model_reproduction.csv", index=False, float_format="%.8f")
    summary.to_csv(args.out / "conformal_sensitivity.csv", index=False, float_format="%.8f")
    pd.DataFrame(regimes).to_csv(args.out / "conformal_by_regime.csv", index=False, float_format="%.8f")
    print(points.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
