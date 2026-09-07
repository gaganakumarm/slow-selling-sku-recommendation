from __future__ import annotations
import argparse
from .config import RAW_DIR, FEATURES_PATH, MODEL_PATH, METRICS_PATH, RECOMMENDATIONS_PATH, DEFAULT_MAX_SERIES
from .data import load_m5_long, make_demo_data
from .features import build_features
from .model import train_and_evaluate
from .recommend import generate_recommendations


def run(demo: bool = False, max_series: int = DEFAULT_MAX_SERIES):
    raw = make_demo_data() if demo else load_m5_long(RAW_DIR, max_series=max_series)
    features = build_features(raw)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURES_PATH, index=False)
    model, metrics, _ = train_and_evaluate(features, MODEL_PATH, METRICS_PATH)
    recs = generate_recommendations(raw, features, model)
    recs.to_csv(RECOMMENDATIONS_PATH, index=False)
    return metrics, recs

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="Run with synthetic smoke-test data")
    p.add_argument("--max-series", type=int, default=DEFAULT_MAX_SERIES)
    args = p.parse_args()
    metrics, recs = run(demo=args.demo, max_series=args.max_series)
    print(metrics)
    print(f"Generated {len(recs)} recommendations -> {RECOMMENDATIONS_PATH}")
