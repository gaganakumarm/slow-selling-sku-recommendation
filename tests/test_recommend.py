import pandas as pd
from src.recommend import choose_discount, risk_from_row


def test_low_risk_gets_no_markdown():
    best, _ = choose_discount("LOW", -1.2, 10.0, 20.0)
    assert best["discount"] == 0.0


def test_fallback_elasticity_caps_high_risk_markdown():
    best, _ = choose_discount("HIGH", -1.0, 10.0, 20.0, "fallback")
    assert 0 <= best["discount"] <= 0.10


def test_high_velocity_decline_is_not_slow_selling_high_risk():
    row = pd.Series({"velocity_percentile": 0.80, "sales_trend_28": -0.55,
                     "rolling_mean_28": 20.0, "forecast_7d": 105.0})
    risk, _, reason = risk_from_row(row)
    assert risk == "LOW"
    assert "velocity remains healthy" in reason[0]


def test_low_velocity_with_strong_decline_is_high_risk():
    row = pd.Series({"velocity_percentile": 0.10, "sales_trend_28": -0.30,
                     "rolling_mean_28": 0.7, "forecast_7d": 2.0})
    risk, score, _ = risk_from_row(row)
    assert risk == "HIGH"
    assert score >= 4
