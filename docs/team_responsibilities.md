# 👥 Team Responsibilities

## Member 1 — Data Engineering Lead

**Goal:** Get clean, usable data from all three modalities into `data/`

### Files to own:
| File | Purpose |
|------|---------|
| `src/data/pubchem_api.py` | Fetch SMILES + chemical data |
| `src/data/drugbank_api.py` | Fetch enzyme/biological data |
| `src/data/openfda_api.py` | Fetch FDA FAERS clinical data |
| `src/data/data_loader.py` | Orchestrate all three sources |
| `notebooks/01_data_exploration.ipynb` | EDA of raw data |

### Step-by-step tasks:
1. Set up `.env` with API keys
2. Run `python src/data/pubchem_api.py` → verify `data/raw/pubchem/compounds.csv`
3. Run `python src/data/drugbank_api.py` → verify enzyme data
4. Run `python src/data/openfda_api.py` → verify FDA data
5. Run `python scripts/download_data.py` → produces `data/processed/interaction_pairs.csv`
6. Do EDA in `notebooks/01_data_exploration.ipynb`

---

## Member 2 — Model Development Lead

**Goal:** Build the multi-modal feature matrix and train all baseline models

### Files to own:
| File | Purpose |
|------|---------|
| `src/features/molecular_features.py` | Morgan fingerprints |
| `src/features/biological_features.py` | Enzyme one-hot vectors |
| `src/features/clinical_features.py` | FDA risk features |
| `src/features/feature_fusion.py` | Combine all 3 modalities |
| `src/models/baseline/` | LR, RF, XGBoost models |
| `src/models/model_trainer.py` | Training pipeline |

### Step-by-step tasks:
1. Wait for Member 1 to produce `data/processed/interaction_pairs.csv`
2. Run `python src/features/molecular_features.py` → test fingerprints
3. Run `python src/features/feature_fusion.py` → test fusion
4. Run `python scripts/train_models.py` → trains all 3 models, saves metrics
5. Check `results/metrics/baseline_metrics.json` for AUROC scores

---

## Member 3 — Explainability & Validation Lead

**Goal:** Make the model's predictions interpretable and clinically validated

### Files to own:
| File | Purpose |
|------|---------|
| `src/explainability/shap_explainer.py` | SHAP values + severity levels |
| `src/explainability/mechanism_extractor.py` | CYP mechanism explanations |
| `notebooks/05_explainability_analysis.ipynb` | SHAP plots + validation |

### Step-by-step tasks:
1. Wait for Member 2 to train and save models
2. Load the XGBoost model and run SHAP on test data
3. Generate `results/figures/shap_summary.png`
4. Test `get_severity()` on range of probabilities
5. Test `MechanismExtractor` on known interacting pairs (warfarin + fluoxetine)
6. Validate mechanism explanations against DrugBank literature

---

## Member 4 — Frontend & Deployment Lead

**Goal:** Build the Streamlit app and connect all modules

### Files to own:
| File | Purpose |
|------|---------|
| `streamlit_app/app.py` | Main app + navigation |
| `streamlit_app/pages/input_medications.py` | Drug pair check page |
| `streamlit_app/pages/batch_analysis.py` | Polypharmacy check page |
| `streamlit_app/pages/risk_dashboard.py` | Data overview page |
| `streamlit_app/pages/model_comparison.py` | Prior work comparison page |
| `src/inference/predictor.py` | Prediction orchestrator |

### Step-by-step tasks:
1. Run app in demo mode (before model is trained): `streamlit run streamlit_app/app.py`
2. Verify each page loads without errors
3. After Member 2 trains models, switch predictor to real model
4. Connect Member 3's SHAP explainer to the prediction output page
5. Deploy to Streamlit Cloud or test locally

---

## 🔁 Development Order

```
Member 1 → Member 2 → Member 3
                ↓
           Member 4 (can start app structure in parallel, connects at end)
```

---

## 🧪 How to Run the Full Pipeline

```bash
# 1. Download all data (Member 1)
python scripts/download_data.py

# 2. Train all models (Member 2)
python scripts/train_models.py

# 3. Launch the app (Member 4)
streamlit run streamlit_app/app.py
```
