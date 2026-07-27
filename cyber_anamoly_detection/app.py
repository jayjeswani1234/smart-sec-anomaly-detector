from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

from src.config import CFG
from src.explain import format_explanation
from src.model import load_bundle, score_events, train_and_save
from src.synthetic_data import generate_synthetic_logs


st.set_page_config(page_title="Cyber Anomaly Detector", layout="wide")
st.title("AI-Powered Behavioral Anomaly Detection for Cybersecurity")

st.caption("Synthetic prototype for SIH demo: generate logs, train model, rank alerts, and explain suspicious activity.")

with st.sidebar:
    st.header("Controls")
    n_entities = st.number_input("Entities", min_value=50, max_value=1000, value=CFG.n_entities, step=50)
    n_events = st.number_input("Events", min_value=500, max_value=50000, value=CFG.n_events, step=500)
    anomaly_rate = st.slider("Anomaly rate", min_value=0.005, max_value=0.15, value=float(CFG.anomaly_rate), step=0.005)
    seed = st.number_input("Seed", min_value=1, max_value=999999, value=CFG.seed, step=1)
    retrain = st.button("Generate + Train")
    load_sample = st.button("Load sample data")

if retrain:
    with st.spinner("Generating synthetic data and training models..."):
        df = generate_synthetic_logs(int(n_entities), int(n_events), float(anomaly_rate), int(seed))
        df.to_csv(Path(CFG.data_dir) / "synthetic_access_logs.csv", index=False)
        train_and_save(seed=int(seed), model_dir=CFG.model_dir)
    st.success("Training complete. Models saved in /models.")

if "bundle" not in st.session_state:
    try:
        st.session_state.bundle = load_bundle()
    except Exception:
        st.session_state.bundle = None

if load_sample or "df" not in st.session_state:
    st.session_state.df = generate_synthetic_logs(int(n_entities), int(n_events), float(anomaly_rate), int(seed))

uploaded = st.file_uploader("Upload access log CSV", type=["csv"])
if uploaded is not None:
    st.session_state.df = pd.read_csv(uploaded)

df = st.session_state.df

if st.session_state.bundle is None:
    st.warning("No trained model found yet. Click **Generate + Train** first or run `python train.py`.")
else:
    scored = score_events(df, st.session_state.bundle)
    scored["explanation"] = scored.apply(format_explanation, axis=1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(scored):,}")
    c2.metric("Anomaly rate", f"{scored['label'].eq('anomaly').mean() * 100:.2f}%")
    c3.metric("Flagged alerts", f"{scored['predicted_anomaly'].sum():,}")
    c4.metric("Top risk", f"{scored['alert_risk'].max():.3f}")

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Top Alerts")
        top_alerts = scored.head(25).copy()
        st.dataframe(
            top_alerts[
                [
                    "timestamp",
                    "entity_id",
                    "entity_type",
                    "geo_location",
                    "resource_accessed",
                    "auth_method",
                    "anomaly_score",
                    "alert_risk",
                    "predicted_attack_type",
                    "explanation",
                ]
            ],
            use_container_width=True,
            height=520,
        )

    with right:
        st.subheader("Alert Distribution")
        chart_df = scored.head(200).copy()
        fig = px.histogram(chart_df, x="alert_risk", nbins=20, title="Risk Score Distribution")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.histogram(scored, x="predicted_attack_type", title="Predicted Attack Types")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Event Details")
    idx = st.number_input("Select row", min_value=0, max_value=max(len(scored)-1, 0), value=0, step=1)
    row = scored.iloc[int(idx)]
    st.json({
        "event_id": row.get("event_id"),
        "entity_id": row.get("entity_id"),
        "label": row.get("label"),
        "predicted_anomaly": bool(row.get("predicted_anomaly")),
        "predicted_attack_type": row.get("predicted_attack_type"),
        "risk": float(row.get("alert_risk")),
        "explanation": row.get("explanation"),
    })

    st.subheader("Raw event")
    st.dataframe(pd.DataFrame([row]), use_container_width=True)
