from pathlib import Path
import pandas as pd
import pytest
from src.data import make_demo_data
from src.features import build_features
from src.model import train_and_evaluate
from src.recommend import generate_recommendations

def test_end_to_end_demo(tmp_path: Path):
    raw = make_demo_data(n_series=5, n_days=160)
    feat = build_features(raw)
    model, metrics, _ = train_and_evaluate(feat, tmp_path/"m.joblib", tmp_path/"metrics.json", holdout_days=21)
    recs = generate_recommendations(raw, feat, model)
    assert len(recs) == 5
    assert {"risk","recommended_discount_pct","risk_reason","elasticity_source"}.issubset(recs.columns)
    assert metrics["model_mae"] >= 0


@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_chronological_28_day_holdout(tmp_path: Path):
    raw = make_demo_data(n_series=2, n_days=100)
    # Align with the verified M5 validation end date.
    raw["date"] += pd.Timestamp("2016-05-22") - raw["date"].max()
    feat = build_features(raw)
    _, metrics, valid = train_and_evaluate(
        feat, tmp_path / "m.joblib", tmp_path / "metrics.json"
    )
    assert metrics["validation_start"] == "2016-04-25"
    assert metrics["validation_days"] == 28
    assert valid["date"].min() == pd.Timestamp("2016-04-25")
    assert valid["date"].max() == pd.Timestamp("2016-05-22")
    assert valid.groupby("id")["date"].nunique().eq(28).all()
    assert metrics["rows_validation"] == 2 * 28
    assert metrics["rows_train"] == len(feat[feat["date"] < pd.Timestamp("2016-04-25")])
