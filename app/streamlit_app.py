import json
import sys
from pathlib import Path

# Streamlit adds app/ to the import path; src/ lives in the repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
import plotly.express as px
from src.config import RECOMMENDATIONS_PATH, METRICS_PATH

st.set_page_config(page_title="Slow-Selling SKU Decision Support", page_icon="📦", layout="wide")
st.title("Slow-Selling SKU Detection & Markdown Recommendations")
st.caption("Applied ML decision-support prototype using M5 sales, price and calendar data. Markdown effects are scenarios, not causal claims or Walmart operational decisions.")

if not RECOMMENDATIONS_PATH.exists():
    st.error("No artifacts found. Run `python -m src.pipeline` after placing M5 files in data/raw/.")
    st.stop()
recs = pd.read_csv(RECOMMENDATIONS_PATH)
metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}

tab1, tab2, tab3 = st.tabs(["Overview", "SKU Analysis", "Model Performance"])
with tab1:
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("SKU-store series", len(recs)); c2.metric("High risk", int((recs.risk=="HIGH").sum()))
    c3.metric("Medium risk", int((recs.risk=="MEDIUM").sum())); c4.metric("Markdown suggested", int((recs.recommended_discount_pct>0).sum()))
    counts = recs["recommended_discount_pct"].value_counts().sort_index().rename_axis("discount_pct").reset_index(name="series")
    counts["discount_pct"] = counts["discount_pct"].map(lambda value: f"{value:g}%")
    fig = px.bar(counts, x="discount_pct", y="series", text="series",
                 title="Recommended Markdown Distribution",
                 labels={"discount_pct": "Recommended markdown", "series": "Number of SKU-store series"})
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=counts["discount_pct"].tolist())
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(recs[["item_id","store_id","risk","velocity_percentile","forecast_7d","sales_trend_28","recommended_discount_pct"]], use_container_width=True, hide_index=True)
with tab2:
    options = (recs.item_id + " | " + recs.store_id).tolist()
    selected = st.selectbox("Select SKU / store", options)
    item, store = [x.strip() for x in selected.split("|")]
    r = recs[(recs.item_id==item)&(recs.store_id==store)].iloc[0]
    a,b,c,d = st.columns(4)
    a.metric("Slow-selling risk", r.risk); b.metric("7-day forecast", f"{r.forecast_7d:.1f} units")
    c.metric("Peer velocity percentile", f"{r.velocity_percentile*100:.0f}%"); d.metric("Recommended markdown", f"{int(r.recommended_discount_pct)}%")
    st.write(f"Item ID: **{item}** | Store ID: **{store}**")
    a,b,c,d = st.columns(4)
    a.metric("Risk score", f"{r.risk_score:g}")
    b.metric("Recent daily sales (28 days)", f"{r.recent_daily_sales_28d:.3f} units")
    c.metric("Recent trend (28 days)", f"{r.sales_trend_28:.2%}")
    d.metric("Current price", f"${r.current_price:.2f}")
    st.subheader("Risk rationale"); st.write(r.risk_reason)
    st.subheader("Markdown rationale"); st.write(r.discount_reason)
    source_label = "historical price/sales association" if r.elasticity_source == "historical_association" else "documented fallback"
    st.write(f"Scenario elasticity: **{r.scenario_elasticity:.2f}** ({source_label}; {int(r.elasticity_weeks)} weekly observations). This parameter supports scenario analysis and is not a causal uplift estimate.")
with tab3:
    if metrics:
        a,b,c,d = st.columns(4)
        a.metric("Baseline MAE", f"{metrics['baseline_mae']:.3f}")
        b.metric("Model MAE", f"{metrics['model_mae']:.3f}")
        c.metric("MAE improvement", f"{metrics['mae_improvement_pct']:.1f}%")
        d.metric("WMAPE", f"{metrics['model_wmape']*100:.1f}%")
        st.json(metrics)
    st.info("Evaluation uses a chronological holdout. Report portfolio metrics only after running on the real M5 data.")
