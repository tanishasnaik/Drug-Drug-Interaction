"""
streamlit_app/app.py
---------------------
Member 4 — Frontend & Deployment Lead

Main Streamlit entry point. Sets up multi-page navigation.

Run with:
    streamlit run streamlit_app/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Drug Interaction Predictor",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ───────────────────────────────────────────
st.sidebar.title("💊 DDI Predictor")
st.sidebar.markdown("**Multi-Modal Explainable AI**")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    options=[
        "🏠 Home",
        "💊 Check Drug Pair",
        "📋 Polypharmacy Check",
        "📊 Risk Dashboard",
        "📈 Model Performance",
    ]
)

st.sidebar.divider()
st.sidebar.caption("⚠️ For research/educational use only. Always consult a physician.")

# ── Route to pages ───────────────────────────────────────────────
if page == "🏠 Home":
    st.title("💊 Drug Interaction Prediction System")
    st.subheader("Multi-Modal Explainable AI for Clinical Safety")

    st.markdown("""
    This system predicts potential drug-drug interactions (DDIs) using:

    | Modality | Source | What it provides |
    |---|---|---|
    | 🧪 Chemical | PubChem | Molecular fingerprints (SMILES) |
    | 🧬 Biological | DrugBank | Enzyme & target overlap |
    | 🏥 Clinical | OpenFDA FAERS | Real-world adverse event data |

    ### 🚀 Our innovations vs prior work:
    - **5-level severity scoring** (Minor → Contraindicated) instead of binary yes/no
    - **Polypharmacy support** — check 3, 5, or 10 drugs at once
    - **Mechanism explanations** — *why* the interaction happens, not just *that* it does
    - **SHAP-based AI explanations** — clinician-interpretable feature importance
    - **Real-world evidence** — FDA adverse event data (none of the prior works used this)
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Prior work coverage", "2 modalities", "We use 3 ✅")
    with col2:
        st.metric("Severity levels", "Binary (2)", "We have 5 ✅")
    with col3:
        st.metric("Polypharmacy", "Pairwise only", "3+ drugs ✅")

elif page == "💊 Check Drug Pair":
    import sys, os
    sys.path.insert(0, os.path.abspath("."))
    from streamlit_app.pages.input_medications import show
    show()

elif page == "📋 Polypharmacy Check":
    import sys, os
    sys.path.insert(0, os.path.abspath("."))
    from streamlit_app.pages.batch_analysis import show
    show()

elif page == "📊 Risk Dashboard":
    import sys, os
    sys.path.insert(0, os.path.abspath("."))
    from streamlit_app.pages.risk_dashboard import show
    show()

elif page == "📈 Model Performance":
    import sys, os
    sys.path.insert(0, os.path.abspath("."))
    from streamlit_app.pages.model_comparison import show
    show()
