"""
streamlit_app/pages/batch_analysis.py
---------------------------------------
Member 4 — Frontend & Deployment Lead

Page 2: Polypharmacy check — enter 3–10 drugs and see all pairwise interactions
displayed as a risk matrix heatmap.

KEY INNOVATION: None of the prior systems support polypharmacy beyond 2 drugs.
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from src.inference.predictor import DDIPredictor


@st.cache_resource
def load_predictor():
    return DDIPredictor()


def show():
    st.title("📋 Polypharmacy Interaction Check")
    st.markdown("""
    **Check interactions across multiple drugs simultaneously.**
    Enter 3–10 drug names, one per line. We'll check all possible pairs
    and display results as a risk matrix.

    > This is a key innovation — prior systems only support pairwise (2-drug) checks.
    """)
    st.divider()

    # ── Drug Input ────────────────────────────────────────────────
    col1, col2 = st.columns([2, 1])
    with col1:
        drugs_text = st.text_area(
            "Enter drugs (one per line)",
            placeholder="warfarin\nfluoxetine\naspirin\nomeprazole",
            height=180,
        )
    with col2:
        st.markdown("**Or upload a CSV**")
        uploaded = st.file_uploader("Upload drug_list.csv", type=["csv"])
        st.markdown("CSV must have a column named `drug`")

    # Parse drugs
    drug_list = []
    if drugs_text:
        drug_list = [d.strip().lower() for d in drugs_text.strip().split("\n") if d.strip()]
    elif uploaded:
        df_upload = pd.read_csv(uploaded)
        if "drug" in df_upload.columns:
            drug_list = df_upload["drug"].str.lower().str.strip().tolist()
        else:
            st.error("CSV must have a column named 'drug'")

    if drug_list:
        st.info(f"**{len(drug_list)} drugs detected:** {', '.join(drug_list)}")

        if len(drug_list) < 2:
            st.warning("Please enter at least 2 drugs.")
            return
        if len(drug_list) > 10:
            st.warning("Maximum 10 drugs supported. Using first 10.")
            drug_list = drug_list[:10]

        from math import comb
        n_pairs = comb(len(drug_list), 2)
        st.caption(f"This will check {n_pairs} drug pairs.")

    analyze_btn = st.button("🔬 Analyze All Interactions", type="primary",
                             disabled=(len(drug_list) < 2))

    if analyze_btn and len(drug_list) >= 2:
        predictor = load_predictor()
        results = []

        progress = st.progress(0, text="Analyzing drug pairs...")
        total = len(drug_list) * (len(drug_list) - 1) // 2

        count = 0
        for i, drug_a in enumerate(drug_list):
            for j, drug_b in enumerate(drug_list):
                if j <= i:
                    continue
                with st.spinner(f"Checking: {drug_a} + {drug_b}"):
                    result = predictor.predict_pair(drug_a, drug_b)
                count += 1
                progress.progress(count / total, text=f"Checking {drug_a} + {drug_b}...")

                if "error" not in result:
                    sev = result["severity"]
                    results.append({
                        "Drug A": drug_a.capitalize(),
                        "Drug B": drug_b.capitalize(),
                        "Risk %": f"{result['probability']*100:.1f}%",
                        "Severity": f"{sev['emoji']} {sev['severity_label']}",
                        "Level": sev["severity_level"],
                        "Shared Enzymes": ", ".join(
                            result["enzyme_overlap"].get("shared_enzymes", [])
                        ) or "None",
                        "Recommendation": sev["recommendation"],
                    })

        progress.empty()

        if not results:
            st.error("Could not fetch data for any drug pairs. Check drug names.")
            return

        df_results = pd.DataFrame(results)

        # ── Risk Matrix Heatmap ───────────────────────────────────
        st.subheader("🗺️ Interaction Risk Matrix")
        matrix = np.zeros((len(drug_list), len(drug_list)))
        drug_labels = [d.capitalize() for d in drug_list]

        for _, row in df_results.iterrows():
            i = drug_labels.index(row["Drug A"])
            j = drug_labels.index(row["Drug B"])
            level = row["Level"]
            matrix[i][j] = level
            matrix[j][i] = level

        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=drug_labels,
            y=drug_labels,
            colorscale=[
                [0.0, "#d4edda"],   # 0 = safe (green)
                [0.2, "#fff3cd"],   # 1 = minor (yellow)
                [0.4, "#ffd27f"],   # 2 = moderate (orange)
                [0.6, "#f8d7da"],   # 3 = major (light red)
                [0.8, "#dc3545"],   # 4 = severe (red)
                [1.0, "#721c24"],   # 5 = contraindicated (dark red)
            ],
            zmin=0, zmax=5,
            text=matrix,
            texttemplate="%{z}",
            colorbar=dict(
                title="Severity Level",
                tickvals=[0, 1, 2, 3, 4, 5],
                ticktext=["None", "Minor", "Moderate", "Major", "Severe", "⛔ Contraindicated"],
            )
        ))
        fig.update_layout(
            title="Drug Interaction Risk Matrix (1–5 severity scale)",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Results Table ─────────────────────────────────────────
        st.subheader("📋 Detailed Results")

        # Sort by severity level descending (most dangerous first)
        df_sorted = df_results.sort_values("Level", ascending=False)
        st.dataframe(
            df_sorted.drop(columns=["Level"]),
            use_container_width=True
        )

        # ── High-risk summary ─────────────────────────────────────
        high_risk = df_results[df_results["Level"] >= 3]
        if not high_risk.empty:
            st.error(f"⚠️ **{len(high_risk)} high-risk interaction(s) detected!**")
            for _, row in high_risk.iterrows():
                st.warning(
                    f"{row['Drug A']} + {row['Drug B']}: "
                    f"{row['Severity']} — {row['Recommendation']}"
                )
        else:
            st.success("✅ No high-risk interactions detected in this drug combination.")

        st.divider()
        st.caption("⚠️ Research/educational use only. Always consult a physician.")
