"""
streamlit_app/pages/input_medications.py
------------------------------------------
Member 4 — Frontend & Deployment Lead

Page 1: Check a single drug pair for interaction risk.
Shows: severity, mechanism explanation, SHAP features, molecule viewer.
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))

import streamlit as st
import base64
from src.inference.predictor import DDIPredictor
from src.utils.chemistry_utils import smiles_to_image_base64


@st.cache_resource
def load_predictor():
    return DDIPredictor()


def show():
    st.title("💊 Check Drug Pair Interaction")
    st.markdown("Enter two drug names to check for potential interactions.")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        drug_a = st.text_input("Drug A", placeholder="e.g. warfarin",
                                help="Type the common or generic drug name")
    with col2:
        drug_b = st.text_input("Drug B", placeholder="e.g. fluoxetine")

    predict_btn = st.button("🔬 Predict Interaction", type="primary",
                             disabled=(not drug_a or not drug_b))

    if predict_btn and drug_a and drug_b:
        predictor = load_predictor()

        with st.spinner(f"Analyzing {drug_a} + {drug_b}..."):
            result = predictor.predict_pair(drug_a.strip(), drug_b.strip())

        if "error" in result:
            st.error(result["error"])
            return

        # ── Demo mode warning ─────────────────────────────────────
        if result.get("demo_mode"):
            st.warning("⚠️ Running in DEMO mode — train the model first for real predictions.")

        # ── Severity Banner ───────────────────────────────────────
        sev = result["severity"]
        severity_colors = {
            "Minor": "success", "Moderate": "warning",
            "Major": "warning", "Severe": "error", "Contraindicated": "error"
        }
        alert_type = severity_colors.get(sev["severity_label"], "info")
        getattr(st, alert_type)(
            f"{sev['emoji']} **{sev['severity_label']} Interaction** "
            f"(Level {sev['severity_level']}/5)  —  "
            f"Probability: {result['probability']:.1%}"
        )

        # ── Key Metrics ───────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Risk Score", f"{result['probability']:.1%}")
        m2.metric("Severity", sev["severity_label"])
        m3.metric("Level", f"{sev['severity_level']} / 5")
        m4.metric("Shared Enzymes",
                  result["enzyme_overlap"].get("enzyme_overlap_count", 0))

        st.progress(result["probability"],
                    text=f"Interaction Risk: {result['probability']:.1%}")

        st.info(f"**Clinical Recommendation:** {sev['recommendation']}")
        st.divider()

        # ── Mechanism Explanation ─────────────────────────────────
        st.subheader("🔬 Mechanism Explanation")
        mech = result["mechanism"]
        st.markdown(f"**Type:** {mech['mechanism_type']}")
        st.markdown(f"**Confidence:** {mech['confidence']}")
        st.info(mech["explanation"])

        shared_enzymes = mech.get("shared_enzymes", [])
        if shared_enzymes:
            st.markdown(f"**Shared Enzyme(s):** `{', '.join(shared_enzymes)}`")
        st.divider()

        # ── Drug Info + Molecule Viewer ───────────────────────────
        st.subheader("🧪 Drug Details")
        dc1, dc2 = st.columns(2)

        for col, drug_info, label in [(dc1, result["drug_a"], drug_a),
                                       (dc2, result["drug_b"], drug_b)]:
            with col:
                st.markdown(f"**{label.capitalize()}**")
                st.write(f"CID: `{drug_info.get('cid', 'N/A')}`")
                st.write(f"Formula: `{drug_info.get('molecular_formula', 'N/A')}`")
                st.write(f"MW: `{drug_info.get('molecular_weight', 'N/A')} g/mol`")

                smiles = drug_info.get("smiles", "")
                if smiles:
                    img_b64 = smiles_to_image_base64(smiles, 280, 180)
                    if img_b64:
                        st.image(
                            base64.b64decode(img_b64),
                            caption=f"{label.capitalize()} structure",
                            use_container_width=True
                        )
        st.divider()

        # ── FDA Evidence ──────────────────────────────────────────
        st.subheader("🏥 Real-World Evidence (FDA FAERS)")
        fc1, fc2 = st.columns(2)
        fc1.metric(f"{drug_a.capitalize()} adverse reports", result["fda_reports_a"])
        fc2.metric(f"{drug_b.capitalize()} adverse reports", result["fda_reports_b"])
        st.caption("Based on recent adverse event reports from FDA FAERS database.")

        st.divider()
        st.caption("⚠️ This tool is for research and educational purposes only.")
