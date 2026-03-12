"""
streamlit_app/pages/model_comparison.py
-----------------------------------------
Member 4 — Frontend & Deployment Lead

Page 4: Model comparison page.
Shows how our multi-modal model compares to prior work and each baseline.
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
from pathlib import Path


def show():
    st.title("📈 Model Performance Comparison")
    st.markdown("""
    Compare our multi-modal approach against baseline models and prior work.
    """)
    st.divider()

    # ── Literature comparison table ───────────────────────────────
    st.subheader("📚 Comparison with Prior Work")

    prior_work = pd.DataFrame([
        {"System": "DeepDDI (2018)",        "Data Modalities": "Chemical only",
         "Severity Levels": 2, "Polypharmacy": "❌", "Explainability": "❌",
         "Real-World Data": "❌", "Live UI": "❌"},
        {"System": "MUFFIN (2021)",         "Data Modalities": "Chemical + Biological",
         "Severity Levels": 2, "Polypharmacy": "❌", "Explainability": "⚠️ Partial",
         "Real-World Data": "❌", "Live UI": "❌"},
        {"System": "KGNN (2020)",           "Data Modalities": "Graph (chem + bio)",
         "Severity Levels": 2, "Polypharmacy": "❌", "Explainability": "❌",
         "Real-World Data": "❌", "Live UI": "❌"},
        {"System": "DDIMDL (2022)",         "Data Modalities": "Chemical + targets",
         "Severity Levels": 2, "Polypharmacy": "❌", "Explainability": "❌",
         "Real-World Data": "⚠️ Partial", "Live UI": "❌"},
        {"System": "DrugBank Checker",      "Data Modalities": "Rules-based",
         "Severity Levels": 3, "Polypharmacy": "⚠️ Up to 5", "Explainability": "⚠️ Partial",
         "Real-World Data": "❌", "Live UI": "✅"},
        {"System": "✅ Our System",          "Data Modalities": "Chemical + Biological + Clinical",
         "Severity Levels": 5, "Polypharmacy": "✅ 3-10 drugs", "Explainability": "✅ SHAP + Mechanism",
         "Real-World Data": "✅ FDA FAERS", "Live UI": "✅ Streamlit"},
    ])

    st.dataframe(
        prior_work.set_index("System"),
        use_container_width=True,
        height=280
    )

    st.divider()

    # ── Our model metrics ─────────────────────────────────────────
    st.subheader("📊 Our Baseline Model Results")
    metrics_path = Path("results/metrics/baseline_metrics.json")

    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

        df_metrics = pd.DataFrame(metrics).T.reset_index()
        df_metrics.columns = ["Model", "AUROC", "F1", "Precision", "Recall"]
        df_metrics["Model"] = df_metrics["Model"].str.replace("_", " ").str.title()

        # Radar chart
        categories = ["AUROC", "F1", "Precision", "Recall"]
        fig = go.Figure()
        for _, row in df_metrics.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[c] for c in categories],
                theta=categories,
                fill="toself",
                name=row["Model"],
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Model Performance Radar Chart",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_metrics, use_container_width=True)

    else:
        st.info("Train the models to see performance metrics here:")
        st.code("python scripts/train_models.py", language="bash")
        st.markdown("""
        **Expected performance** (from literature benchmarks with similar approaches):

        | Model | Expected AUROC | Expected F1 |
        |---|---|---|
        | Logistic Regression | 0.72–0.78 | 0.65–0.71 |
        | Random Forest | 0.82–0.87 | 0.75–0.82 |
        | XGBoost | 0.85–0.90 | 0.78–0.85 |
        | Multi-modal XGBoost | **0.88–0.93** | **0.82–0.88** |
        """)
