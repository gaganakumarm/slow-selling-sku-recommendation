from __future__ import annotations
import numpy as np
import pandas as pd
from .features import FEATURE_COLUMNS


def estimate_price_elasticity(history: pd.DataFrame) -> tuple[float, str, int]:
    """Return scenario elasticity, provenance, and usable weekly observations.

    This is descriptive scenario analysis, not a causal estimate of markdown uplift.
    """
    weekly = (history.set_index("date").resample("7D")
              .agg(sales=("sales", "sum"), price=("sell_price", "mean")).dropna())
    usable = int(len(weekly))
    if usable < 8 or weekly["price"].nunique() < 3:
        return -1.0, "fallback", usable
    x = np.log(weekly["price"].clip(lower=0.01)).to_numpy()
    y = np.log1p(weekly["sales"]).to_numpy()
    slope = float(np.polyfit(x, y, 1)[0])
    # Positive slopes are economically implausible for this simple scenario model;
    # use a neutral documented fallback rather than presenting them as learned facts.
    if not np.isfinite(slope) or slope >= 0:
        return -1.0, "fallback", usable
    return float(np.clip(slope, -3.0, -0.2)), "historical_association", usable


def risk_from_row(row: pd.Series) -> tuple[str, int, list[str]]:
    """Classify slow-selling risk using velocity first, deterioration second.

    A high-volume SKU is not labelled slow-selling solely because sales declined.
    velocity_percentile is calculated within category/store peers (lower = slower).
    """
    vp = float(row["velocity_percentile"])
    trend = float(row["sales_trend_28"])
    recent_week = float(row["rolling_mean_28"]) * 7
    forecast = float(row["forecast_7d"])

    very_low_velocity = vp <= 0.15
    low_velocity = vp <= 0.35
    strong_decline = trend <= -0.20
    moderate_decline = trend <= -0.05
    forecast_deterioration = forecast < max(1.0, 0.75 * recent_week)

    score, reasons = 0, []
    if very_low_velocity:
        score += 2; reasons.append("sales velocity is in the bottom 15% of category/store peers")
    elif low_velocity:
        score += 1; reasons.append("sales velocity is in the bottom 35% of category/store peers")

    # Deterioration contributes only when the SKU is already relatively slow.
    if low_velocity and strong_decline:
        score += 2; reasons.append("recent sales trend is declining strongly")
    elif low_velocity and moderate_decline:
        score += 1; reasons.append("recent sales trend is declining")
    if low_velocity and forecast_deterioration:
        score += 1; reasons.append("near-term forecast is below its recent demand rate")

    risk = "HIGH" if score >= 4 else "MEDIUM" if score >= 2 else "LOW"
    if risk == "LOW" and strong_decline and not low_velocity:
        reasons = ["sales are declining, but velocity remains healthy relative to category/store peers"]
    elif not reasons:
        reasons = ["sales velocity and near-term demand do not indicate slow-selling risk"]
    return risk, score, reasons


def choose_discount(risk: str, elasticity: float, price: float, forecast_units: float,
                    elasticity_source: str = "historical_association"):
    """Choose the smallest useful markdown under conservative guardrails."""
    if risk == "LOW":
        allowed = (0.0,)
    elif risk == "MEDIUM":
        allowed = (0.0, 0.05, 0.10)
    else:
        allowed = (0.0, 0.05, 0.10, 0.15, 0.20)

    # When elasticity is a fallback, cap action size: uncertainty should make the
    # decision more conservative, not more aggressive.
    if elasticity_source == "fallback":
        cap = 0.05 if risk == "MEDIUM" else 0.10 if risk == "HIGH" else 0.0
        allowed = tuple(d for d in allowed if d <= cap)

    candidates = []
    for d in allowed:
        demand_multiplier = (1 - d) ** elasticity
        units = max(0.0, forecast_units * demand_multiplier)
        revenue = units * price * (1 - d)
        candidates.append({"discount": d, "expected_units_7d": units, "expected_revenue_7d": revenue})

    base = candidates[0]
    base_units = max(base["expected_units_7d"], 1e-9)
    base_rev = max(base["expected_revenue_7d"], 1e-9)
    # A markdown must provide >=5% scenario sell-through uplift and retain >=95%
    # of baseline revenue. Select the smallest qualifying action.
    feasible = [c for c in candidates[1:]
                if c["expected_units_7d"] >= 1.05 * base_units
                and c["expected_revenue_7d"] >= 0.95 * base_rev]
    best = min(feasible, key=lambda c: c["discount"]) if feasible else base
    return best, candidates


def generate_recommendations(history: pd.DataFrame, features: pd.DataFrame, model) -> pd.DataFrame:
    latest = features.sort_values("date").groupby("id", as_index=False).tail(1).copy()
    latest["forecast_daily"] = np.clip(model.predict(latest[FEATURE_COLUMNS]), 0, None)
    latest["forecast_7d"] = latest["forecast_daily"] * 7
    # Percentile rank within comparable category/store peers. pct=True gives
    # smaller values to slower sellers, which is what the risk rules consume.
    latest["velocity_percentile"] = latest.groupby(["cat_id", "store_id"])["rolling_mean_28"].rank(pct=True, method="average")

    records = []
    for _, row in latest.iterrows():
        hist = history[history["id"] == row["id"]]
        elasticity, elasticity_source, elasticity_weeks = estimate_price_elasticity(hist)
        risk, score, reasons = risk_from_row(row)
        best, _ = choose_discount(risk, elasticity, float(row["sell_price"]), float(row["forecast_7d"]), elasticity_source)
        discount_reason = ("no markdown: slow-selling risk is low" if risk == "LOW" else
                           "no markdown: no candidate met the 5% unit-uplift and 95% revenue guardrails" if best["discount"] == 0 else
                           "smallest candidate meeting the scenario uplift and revenue guardrails")
        records.append({
            "id": row["id"], "item_id": row["item_id"], "store_id": row["store_id"],
            "category": row["cat_id"], "risk": risk, "risk_score": score,
            "velocity_percentile": round(float(row["velocity_percentile"]), 4),
            "forecast_7d": round(float(row["forecast_7d"]), 3),
            "recent_daily_sales_28d": round(float(row["rolling_mean_28"]), 3),
            "sales_trend_28": round(float(row["sales_trend_28"]), 4),
            "current_price": round(float(row["sell_price"]), 2),
            "scenario_elasticity": round(elasticity, 3),
            "elasticity_source": elasticity_source,
            "elasticity_weeks": elasticity_weeks,
            "recommended_discount_pct": int(round(best["discount"] * 100)),
            "expected_units_7d": round(best["expected_units_7d"], 3),
            "expected_revenue_7d": round(best["expected_revenue_7d"], 2),
            "risk_reason": "; ".join(reasons),
            "discount_reason": discount_reason,
        })
    return pd.DataFrame(records).sort_values(["risk_score", "velocity_percentile", "sales_trend_28"], ascending=[False, True, True])
