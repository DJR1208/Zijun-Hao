
"""
CFPS household consumption dynamics model comparison
- Training period: 2010-2016 (med / eec)
- Forecast period: 2018, 2020, 2022
- Models: random walk, jump diffusion (symmetric 2σ, asymmetric positive 1σ, symmetric 1σ), mean reversion
- Evaluation: aggregate forecast errors across 192 households and compare total RMSE
- Runs separately for med, eec, and combined (med + eec)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ========== Configuration ==========
EXCEL_PATH = Path(
   "/Users/joanna/Desktop/Pioneer/cfps筛选.xlsx"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
TRAIN_YEARS = [2010, 2012, 2014, 2016]
TEST_YEARS = [2018, 2020, 2022]
ALL_YEARS = TRAIN_YEARS + TEST_YEARS
CONSUMPTION_COLS = ["med", "eec"]  # med: medical expenditure; eec: culture & entertainment expenditure
DT = 2  # CFPS is biennial; consecutive observations are 2 years apart


@dataclass
class ModelParams:
    model_name: str
    params: Dict[str, float]


@dataclass
class ForecastResult:
    household_id: float
    consumption_type: str
    year: int
    actual: float
    pred_rw: float
    pred_jump_sym_2sigma: float    # symmetric jumps, 2σ threshold
    pred_jump_asym_1sigma: float   # asymmetric positive jumps, 1σ threshold
    pred_jump_sym_1sigma: float    # symmetric jumps, 1σ threshold (control)
    pred_mr: float


# ========== Data loading ==========
def load_panel_data(path: Path) -> pd.DataFrame:
    """Load Excel file and reshape into household-year panel data."""
    df = pd.read_excel(path)

    id_cols = ["fid10", "fid12", "fid14", "fid16", "fid18", "fid20", "fid22"]
    df["household_id"] = df[id_cols].bfill(axis=1).iloc[:, 0]

    panel = (
        df[["household_id", "year"] + CONSUMPTION_COLS]
        .copy()
        .dropna(subset=["year"])
    )
    panel["year"] = panel["year"].astype(int)
    panel = panel[panel["year"].isin(ALL_YEARS)]
    panel = panel.sort_values(["household_id", "year"]).reset_index(drop=True)

    dup = panel.duplicated(["household_id", "year"], keep=False)
    if dup.any():
        panel = panel.groupby(["household_id", "year"], as_index=False)[CONSUMPTION_COLS].mean()

    return panel


# ========== Model parameter estimation (2010-2016) ==========
def _safe_std(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def fit_random_walk(series: np.ndarray) -> ModelParams:
    """
    Random walk with drift:
        X_t = X_{t-1} + mu + epsilon_t
    Each 2-year interval is one time step.
    """
    diffs = np.diff(series)
    mu = float(np.mean(diffs)) if len(diffs) else 0.0
    sigma = _safe_std(diffs)
    return ModelParams("random_walk", {"mu": mu, "sigma": sigma})


def fit_mean_reversion(series: np.ndarray) -> ModelParams:
    """
    Mean reversion (discrete OU / AR(1) form):
        X_t - X_{t-1} = kappa * (theta - X_{t-1}) * DT + sigma * sqrt(DT) * epsilon
    Equivalent to OLS of delta X on X_{t-1}.
    """
    if len(series) < 2:
        return ModelParams("mean_reversion", {"kappa": 0.0, "theta": float(series[-1]), "sigma": 0.0})

    x_lag = series[:-1]
    delta_x = np.diff(series)

    if np.allclose(x_lag, x_lag[0]):
        theta = float(np.mean(series))
        sigma = _safe_std(delta_x)
        return ModelParams("mean_reversion", {"kappa": 0.0, "theta": theta, "sigma": sigma})

    # delta_x = alpha + beta * x_lag
    beta, alpha = np.polyfit(x_lag, delta_x, 1)
    kappa = max(-beta / DT, 0.0)
    theta = alpha / (kappa * DT) if kappa > 1e-8 else float(np.mean(series))

    fitted = alpha + beta * x_lag
    resid = delta_x - fitted
    sigma = _safe_std(resid) / np.sqrt(DT) if len(resid) > 1 else 0.0

    return ModelParams("mean_reversion", {"kappa": kappa, "theta": theta, "sigma": sigma})


def fit_jump_diffusion(series: np.ndarray,
                       mode: str = "symmetric",
                       threshold_sigma: float = 2.0) -> ModelParams:
    """
    Jump diffusion (Merton-type, discrete approximation).

    Parameters:
        mode: "symmetric" → both positive and negative extreme changes are jumps
              "asymmetric" → only positive extreme changes are jumps
        threshold_sigma: jump detection threshold in multiples of standard deviation
    """
    model_name = f"jump_diffusion_{mode}_{threshold_sigma}sigma"

    if len(series) < 2:
        last = float(series[-1])
        return ModelParams(
            model_name,
            {"mu": 0.0, "sigma": 0.0, "lambda": 0.0, "mu_j": 0.0, "sigma_j": 0.0, "last_level": last},
        )

    diffs = np.diff(series)
    mu = float(np.mean(diffs)) / DT
    sigma = _safe_std(diffs) / np.sqrt(DT) if len(diffs) > 1 else 0.0

    threshold = threshold_sigma * sigma * np.sqrt(DT)

    # Detect jumps based on mode
    if mode == "symmetric":
        # Both positive and negative extremes are jumps
        jump_mask = np.abs(diffs - mu * DT) > threshold if threshold > 0 else np.zeros(len(diffs), dtype=bool)
    else:
        # Only positive extreme changes are jumps
        jump_mask = (diffs - mu * DT) > threshold if threshold > 0 else np.zeros(len(diffs), dtype=bool)

    jump_sizes = diffs[jump_mask]
    diff_sizes = diffs[~jump_mask]

    if len(diff_sizes) >= 1:
        mu = float(np.mean(diff_sizes)) / DT
        sigma = _safe_std(diff_sizes) / np.sqrt(DT) if len(diff_sizes) > 1 else sigma

    total_years = DT * len(diffs)
    lam = len(jump_sizes) / total_years if total_years > 0 else 0.0
    mu_j = float(np.mean(jump_sizes)) if len(jump_sizes) else 0.0
    sigma_j = _safe_std(jump_sizes) if len(jump_sizes) > 1 else 0.0

    return ModelParams(
        model_name,
        {"mu": mu, "sigma": sigma, "lambda": lam, "mu_j": mu_j, "sigma_j": sigma_j},
    )


# ========== Multi-step forecasting (from 2016) ==========
def predict_random_walk(last_value: float, params: Dict[str, float], horizon_years: int) -> float:
    """horizon_years: years ahead from 2016 (2/4/6)."""
    steps = horizon_years / DT
    return last_value + params["mu"] * steps


def predict_mean_reversion(last_value: float, params: Dict[str, float], horizon_years: int) -> float:
    kappa = params["kappa"]
    theta = params["theta"]
    if kappa < 1e-8:
        return last_value
    return theta + (last_value - theta) * np.exp(-kappa * horizon_years)


def predict_jump_diffusion(last_value: float, params: Dict[str, float], horizon_years: int) -> float:
    drift = params["mu"] + params["lambda"] * params["mu_j"]
    return last_value + drift * horizon_years


def forecast_from_2016(
    last_value: float,
    rw_params: Dict[str, float],
    jump_sym_2sigma_params: Dict[str, float],
    jump_asym_1sigma_params: Dict[str, float],
    jump_sym_1sigma_params: Dict[str, float],
    mr_params: Dict[str, float],
    target_year: int,
) -> Tuple[float, float, float, float, float]:
    horizon = target_year - 2016
    pred_rw = predict_random_walk(last_value, rw_params, horizon)
    pred_jump_sym_2sigma = predict_jump_diffusion(last_value, jump_sym_2sigma_params, horizon)
    pred_jump_asym_1sigma = predict_jump_diffusion(last_value, jump_asym_1sigma_params, horizon)
    pred_jump_sym_1sigma = predict_jump_diffusion(last_value, jump_sym_1sigma_params, horizon)
    pred_mr = predict_mean_reversion(last_value, mr_params, horizon)
    return (
        max(pred_rw, 0.0),
        max(pred_jump_sym_2sigma, 0.0),
        max(pred_jump_asym_1sigma, 0.0),
        max(pred_jump_sym_1sigma, 0.0),
        max(pred_mr, 0.0),
    )


# ========== Main workflow ==========
def run_analysis(
    panel: pd.DataFrame,
    consumption_types: List[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    cols = consumption_types if consumption_types is not None else CONSUMPTION_COLS
    param_records: List[dict] = []
    forecast_records: List[ForecastResult] = []

    for hh_id, hh_df in panel.groupby("household_id"):
        hh_df = hh_df.set_index("year")

        for col in cols:
            train = hh_df.loc[hh_df.index.isin(TRAIN_YEARS), col].dropna()
            if len(train) < 2 or 2016 not in train.index:
                continue

            train_values = train.loc[sorted(train.index)].to_numpy(dtype=float)
            last_value = float(train.loc[2016])

            # Estimate all model parameters
            rw = fit_random_walk(train_values)
            jump_sym_2sigma = fit_jump_diffusion(train_values, mode="symmetric", threshold_sigma=2.0)
            jump_asym_1sigma = fit_jump_diffusion(train_values, mode="asymmetric", threshold_sigma=1.0)
            jump_sym_1sigma = fit_jump_diffusion(train_values, mode="symmetric", threshold_sigma=1.0)
            mr = fit_mean_reversion(train_values)

            # Save parameter records
            for model, params in [
                (rw, rw.params),
                (jump_sym_2sigma, jump_sym_2sigma.params),
                (jump_asym_1sigma, jump_asym_1sigma.params),
                (jump_sym_1sigma, jump_sym_1sigma.params),
                (mr, mr.params),
            ]:
                rec = {
                    "household_id": hh_id,
                    "consumption_type": col,
                    "model": model.model_name,
                }
                rec.update(params)
                param_records.append(rec)

            # Generate predictions
            for test_year in TEST_YEARS:
                if test_year not in hh_df.index or pd.isna(hh_df.loc[test_year, col]):
                    continue

                actual = float(hh_df.loc[test_year, col])
                pred_rw, pred_jump_sym_2sigma, pred_jump_asym_1sigma, pred_jump_sym_1sigma, pred_mr = \
                    forecast_from_2016(
                        last_value,
                        rw.params,
                        jump_sym_2sigma.params,
                        jump_asym_1sigma.params,
                        jump_sym_1sigma.params,
                        mr.params,
                        test_year,
                    )
                forecast_records.append(
                    ForecastResult(
                        hh_id, col, test_year, actual,
                        pred_rw,
                        pred_jump_sym_2sigma,
                        pred_jump_asym_1sigma,
                        pred_jump_sym_1sigma,
                        pred_mr,
                    )
                )

    params_df = pd.DataFrame(param_records)
    forecast_df = pd.DataFrame([f.__dict__ for f in forecast_records])

    if forecast_df.empty:
        raise ValueError("No forecast samples available for evaluation; check missing data.")

    # Compute prediction errors for each model
    forecast_df["err_rw"] = forecast_df["actual"] - forecast_df["pred_rw"]
    forecast_df["err_jump_sym_2sigma"] = forecast_df["actual"] - forecast_df["pred_jump_sym_2sigma"]
    forecast_df["err_jump_asym_1sigma"] = forecast_df["actual"] - forecast_df["pred_jump_asym_1sigma"]
    forecast_df["err_jump_sym_1sigma"] = forecast_df["actual"] - forecast_df["pred_jump_sym_1sigma"]
    forecast_df["err_mr"] = forecast_df["actual"] - forecast_df["pred_mr"]

    def rmse(errors: pd.Series) -> float:
        return float(np.sqrt(np.mean(errors ** 2)))

    rmse_values = {
        "random_walk": rmse(forecast_df["err_rw"]),
        "jump_diffusion_symmetric_2sigma": rmse(forecast_df["err_jump_sym_2sigma"]),
        "jump_diffusion_asymmetric_1sigma": rmse(forecast_df["err_jump_asym_1sigma"]),
        "jump_diffusion_symmetric_1sigma": rmse(forecast_df["err_jump_sym_1sigma"]),
        "mean_reversion": rmse(forecast_df["err_mr"]),
    }
    rmse_series = pd.Series(rmse_values, name="total_rmse")
    best_model = rmse_series.idxmin()

    rmse_df = rmse_series.to_frame().T
    rmse_df["best_model"] = best_model

    return params_df, forecast_df, rmse_df, rmse_series


def print_report(
    rmse: pd.Series,
    forecast_df: pd.DataFrame,
    consumption_label: str = "All Consumption Types",
) -> None:
    n_hh = forecast_df["household_id"].nunique()
    n_obs = len(forecast_df)

    model_labels = {
        "random_walk": "Random Walk",
        "jump_diffusion_symmetric_2sigma": "Jump Diffusion (Symmetric, 2σ)",
        "jump_diffusion_asymmetric_1sigma": "Jump Diffusion (Asymmetric Positive, 1σ)",
        "jump_diffusion_symmetric_1sigma": "Jump Diffusion (Symmetric, 1σ)",
        "mean_reversion": "Mean Reversion",
    }

    print("=" * 60)
    print("CFPS Consumption Dynamics Model Comparison")
    print(f"Consumption type: {consumption_label}")
    print("=" * 60)
    print(f"Households evaluated: {n_hh}")
    print(f"Forecast samples (household x year): {n_obs}")
    print()
    print("Total RMSE by model (lower is better):")
    for model in [
        "random_walk",
        "jump_diffusion_symmetric_2sigma",
        "jump_diffusion_asymmetric_1sigma",
        "jump_diffusion_symmetric_1sigma",
        "mean_reversion",
    ]:
        print(f"  {model_labels[model]:40s} ({model}): {rmse[model]:,.2f}")

    best = rmse.idxmin()
    print()
    print(f"Best model: {model_labels[best]} ({best}), total RMSE = {rmse[best]:,.2f}")
    print("=" * 60)


def save_results(
    params_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    rmse_df: pd.DataFrame,
    suffix: str,
) -> None:
    """Save analysis outputs with a filename suffix (e.g., med, eec, combined)."""
    params_df.to_csv(
        OUTPUT_DIR / f"model_parameters_{suffix}.csv", index=False, encoding="utf-8-sig"
    )
    forecast_df.to_csv(
        OUTPUT_DIR / f"forecasts_vs_actual_{suffix}.csv", index=False, encoding="utf-8-sig"
    )
    rmse_df.to_csv(
        OUTPUT_DIR / f"rmse_summary_{suffix}.csv", index=False, encoding="utf-8-sig"
    )


def run_and_report(
    panel: pd.DataFrame,
    consumption_types: List[str],
    label: str,
    suffix: str,
) -> pd.Series:
    """Run one model comparison for the given consumption type(s) and save outputs."""
    params_df, forecast_df, rmse_df, rmse_series = run_analysis(panel, consumption_types)
    print_report(rmse_series, forecast_df, consumption_label=label)
    save_results(params_df, forecast_df, rmse_df, suffix)
    print()
    return rmse_series


def main() -> None:
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {EXCEL_PATH}")

    print(f"Loading data: {EXCEL_PATH}")
    panel = load_panel_data(EXCEL_PATH)
    print(f"Panel data: {panel['household_id'].nunique()} households, {len(panel)} records")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    analysis_tasks = [
        (["med"], "Medical Expenditure (med)", "med"),
        (["eec"], "Culture & Entertainment Expenditure (eec)", "eec"),
        (CONSUMPTION_COLS, "Combined (med + eec)", "combined"),
    ]

    for consumption_types, label, suffix in analysis_tasks:
        run_and_report(panel, consumption_types, label, suffix)

    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
