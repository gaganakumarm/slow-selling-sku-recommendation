from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "demand_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
RECOMMENDATIONS_PATH = PROCESSED_DIR / "recommendations.csv"
FEATURES_PATH = PROCESSED_DIR / "features.csv"

RANDOM_STATE = 42
FORECAST_HORIZON_DAYS = 28
DEFAULT_MAX_SERIES = 500
DISCOUNT_LEVELS = (0.0, 0.05, 0.10, 0.15, 0.20)
