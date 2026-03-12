"""
streamlit_app/pages/risk_dashboard.py
---------------------------------------
Member 4 — Frontend & Deployment Lead

Page 3: Risk dashboard showing system-level statistics and drug interaction landscape.
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))

import streamlit as st
import pandas as pd
import plotly.express as px
import json
from pathlib import Path


def show():
    st.title("📊 Risk Dashboard")
    st.markdown("System-level overview of drug interactions and model insights.")
    st.divider()

    # ── Load results if available ─────────────────────────────────
    pairs_path = Path("data/processed/interaction_pairs.csv")
    metrics_path = Path("results/metrics/baseline_metrics.json")

    if pairs_path.exists():
        df = pd.read_csv(pairs_path)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Drug Pairs", len(df))
        col2.metric("Interactions Detected", int(df["label"].sum()))
        col3.metric("Unique Drugs", len(set(df["drug_a"]) | set(df["drug_b"])))
        col4.metric("Interaction Rate",
                    f"{df['label'].mean()*100:.1f}%")

        st.subheader("Enzyme Overlap Distribution")
        if "enzyme_overlap" in df.columns:
            fig = px.histogram(
                df, x="enzyme_overlap",
                color="label",
                color_discrete_map={0: "#28a745", 1: "#dc3545"},
                labels={"enzyme_overlap": "Shared Enzyme Count", "label": "Interaction"},
                title="Enzyme Overlap vs Interaction Label",
                barmode="overlay"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("FDA Risk Score Distribution")
        if "fda_score" in df.columns:
            fig2 = px.box(
                df, x="label", y="fda_score",
                color="label",
                color_discrete_map={0: "#28a745", 1: "#dc3545"},
                labels={"label": "Interaction (0=No, 1=Yes)", "fda_score": "FDA Risk Score"},
                title="FDA FAERS Risk Score by Interaction Label"
            )
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No data found yet. Run the data ingestion pipeline first:")
        st.code("python src/data/data_loader.py", language="bash")

    st.divider()

    # ── Model metrics ─────────────────────────────────────────────
    st.subheader("📈 Model Performance")
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

        df_metrics = pd.DataFrame(metrics).T.reset_index()
        df_metrics.columns = ["Model", "AUROC", "F1", "Precision", "Recall"]
        st.dataframe(df_metrics, use_container_width=True)

        fig3 = px.bar(
            df_metrics.melt(id_vars="Model", var_name="Metric", value_name="Score"),
            x="Model", y="Score", color="Metric",
            barmode="group",
            title="Baseline Model Comparison"
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No model metrics yet. Train the models first:")
        st.code("python scripts/train_models.py", language="bash")
