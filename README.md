# 💊 Drug Interaction Prediction using Multi-Modal Explainable AI

> Predicts drug-drug interactions using chemical, biological, and real-world clinical data — with severity scoring and mechanism-based explanations.

---

## 🧠 What makes this different from prior work?

| Feature | DeepDDI | MUFFIN | KGNN | DrugBank | **Ours** |
|---|---|---|---|---|---|
| Data Modalities | Chemical | Chem + Bio | Graph | Rules | **Chem + Bio + Clinical** |
| Severity Levels | Binary | Binary | Binary | 3 levels | **5 levels** |
| Polypharmacy (3+ drugs) | ❌ | ❌ | ❌ | ⚠️ | **✅** |
| Real-World FDA Data | ❌ | ❌ | ❌ | ❌ | **✅ FAERS** |
| Explainability | ❌ | ⚠️ | ❌ | ⚠️ | **✅ SHAP + Mechanism** |
| Live UI | ❌ | ❌ | ❌ | ✅ | **✅ Streamlit** |
| Continuous Learning | ❌ | ❌ | ❌ | ❌ | **✅ Monthly retraining** |

---

## 🏗️ Project Structure

```
drug-interaction-prediction/
├── config/                    ← All settings (YAML)
├── data/                      ← Raw, processed, interim data
├── docs/                      ← Documentation
│   └── team_responsibilities.md
├── models/                    ← Saved model files
├── notebooks/                 ← Jupyter EDA notebooks
├── results/                   ← Metrics, figures, predictions
├── scripts/                   ← Run pipeline steps
│   ├── download_data.py       ← Step 1: fetch all data
│   ├── train_models.py        ← Step 2: train models
│   └── continuous_learning.py ← Monthly retraining
├── src/
│   ├── data/                  ← API fetchers (Member 1)
│   │   ├── pubchem_api.py
│   │   ├── drugbank_api.py
│   │   └── openfda_api.py
│   ├── features/              ← Feature engineering (Member 2)
│   │   ├── molecular_features.py
│   │   ├── biological_features.py
│   │   ├── clinical_features.py
│   │   └── feature_fusion.py
│   ├── models/baseline/       ← ML models (Member 2)
│   ├── explainability/        ← SHAP + mechanisms (Member 3)
│   └── inference/             ← Prediction engine (Member 4)
└── streamlit_app/             ← Web interface (Member 4)
    ├── app.py
    └── pages/
```

---

## ⚙️ Setup

```bash
# 1. Clone and enter the project
git clone https://github.com/YOUR_USERNAME/drug-interaction-prediction.git
cd drug-interaction-prediction

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# Edit .env — add your DrugBank API key (optional but recommended)
```

---

## 🚀 Run the Pipeline

```bash
# Step 1: Download all data from APIs
python scripts/download_data.py

# Step 2: Train all models
python scripts/train_models.py

# Step 3: Launch the Streamlit app
streamlit run streamlit_app/app.py
```

---

## 📊 Data Sources

| Source | What we get | Key |
|---|---|---|
| PubChem | SMILES, molecular formula | Free |
| DrugBank | Enzymes, targets, DDI pairs | Free account |
| OpenFDA | FAERS adverse event reports | Free (optional key) |

---

## ⚠️ Disclaimer

This system is for **research and educational purposes only**.  
Always consult a licensed healthcare provider for medical decisions.

---

## 📄 License

MIT License
