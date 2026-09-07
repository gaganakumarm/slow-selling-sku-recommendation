from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .features import FEATURE_COLUMNS
from .config import RANDOM_STATE


def wmape(y_true, y_pred) -> float:
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / denom) if denom else 0.0


def train_and_evaluate(features: pd.DataFrame, model_path: Path, metrics_path: Path, holdout_days: int = 28):
    cutoff = features["date"].max() - pd.Timedelta(int(holdout_days) - 1, unit="D")
    train = features[features["date"] < cutoff]
    valid = features[features["date"] >= cutoff]
    if train.empty or valid.empty:
        raise ValueError("Not enough history for temporal train/validation split")
    Xtr, ytr = train[FEATURE_COLUMNS], train["sales"].astype(float)
    Xv, yv = valid[FEATURE_COLUMNS], valid["sales"].astype(float)

    baseline = valid["rolling_mean_7"].clip(lower=0).to_numpy()
    model = HistGradientBoostingRegressor(
        learning_rate=0.07, max_iter=180, max_leaf_nodes=31,
        l2_regularization=0.1, random_state=RANDOM_STATE
    )
    model.fit(Xtr, ytr)
    pred = np.clip(model.predict(Xv), 0, None)
    metrics = {
        "validation_start": str(cutoff.date()),
        "validation_days": holdout_days,
        "rows_train": int(len(train)), "rows_validation": int(len(valid)),
        "baseline_mae": float(mean_absolute_error(yv, baseline)),
        "model_mae": float(mean_absolute_error(yv, pred)),
        "model_rmse": float(mean_squared_error(yv, pred) ** 0.5),
        "model_wmape": wmape(yv.to_numpy(), pred),
    }
    metrics["mae_improvement_pct"] = 100 * (metrics["baseline_mae"] - metrics["model_mae"]) / max(metrics["baseline_mae"], 1e-9)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    valid_out = valid[["id", "item_id", "store_id", "date", "sales", "rolling_mean_7"]].copy()
    valid_out["prediction"] = pred
    return model, metrics, valid_out
