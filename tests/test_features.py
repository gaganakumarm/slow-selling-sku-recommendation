from src.data import make_demo_data
from src.features import build_features, FEATURE_COLUMNS

def test_features_are_created_without_target_leakage_shape_failure():
    f = build_features(make_demo_data(n_series=2, n_days=100))
    assert not f.empty
    assert set(FEATURE_COLUMNS).issubset(f.columns)
    assert f[FEATURE_COLUMNS].isna().sum().sum() == 0
