import json
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from src.config import RECOMMENDATIONS_PATH, METRICS_PATH

app = FastAPI(title="Slow-Selling SKU Decision Support API", version="1.1.0")

def _recs():
    if not RECOMMENDATIONS_PATH.exists():
        raise HTTPException(503, "Artifacts not found. Run: python -m src.pipeline")
    return pd.read_csv(RECOMMENDATIONS_PATH)

@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/metrics")
def metrics():
    if not METRICS_PATH.exists(): raise HTTPException(503, "Metrics not found")
    return json.loads(METRICS_PATH.read_text())

@app.get("/recommendations")
def recommendations(risk: str | None = Query(default=None), limit: int = Query(50, ge=1, le=500)):
    df = _recs()
    if risk: df = df[df["risk"] == risk.upper()]
    return df.head(limit).to_dict(orient="records")

@app.get("/recommendations/{item_id}/{store_id}")
def recommendation(item_id: str, store_id: str):
    df = _recs()
    hit = df[(df.item_id == item_id) & (df.store_id == store_id)]
    if hit.empty: raise HTTPException(404, "Recommendation not found")
    return hit.iloc[0].to_dict()
