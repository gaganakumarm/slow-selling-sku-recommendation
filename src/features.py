from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "lag_7", "lag_14", "lag_28", "rolling_mean_7", "rolling_mean_28",
    "rolling_std_28", "sales_trend_28", "sell_price", "price_change_pct",
    "wday", "month", "event", "snap"
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(["id", "date"])
    g = out.groupby("id", group_keys=False)
    out["lag_7"] = g["sales"].shift(7)
    out["lag_14"] = g["sales"].shift(14)
    out["lag_28"] = g["sales"].shift(28)
    shifted = g["sales"].shift(1)
    out["rolling_mean_7"] = shifted.groupby(out["id"]).transform(lambda s: s.rolling(7).mean())
    out["rolling_mean_28"] = shifted.groupby(out["id"]).transform(lambda s: s.rolling(28).mean())
    out["rolling_std_28"] = shifted.groupby(out["id"]).transform(lambda s: s.rolling(28).std())
    mean_recent = shifted.groupby(out["id"]).transform(lambda s: s.rolling(14).mean())
    mean_prior = g["sales"].shift(15).groupby(out["id"]).transform(lambda s: s.rolling(14).mean())
    out["sales_trend_28"] = (mean_recent - mean_prior) / (mean_prior.abs() + 1.0)
    out["sell_price"] = out.groupby("id")["sell_price"].transform(lambda s: s.ffill().bfill())
    out["price_change_pct"] = out.groupby("id")["sell_price"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    for c in ["wday", "month", "event", "snap"]:
        if c not in out:
            out[c] = 0
    return out.dropna(subset=FEATURE_COLUMNS + ["sales"]).reset_index(drop=True)
