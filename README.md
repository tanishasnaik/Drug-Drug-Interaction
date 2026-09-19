# Drug Interaction Prediction Using Multi-Modal Explainable AI

A Python-based machine learning system for predicting drug-drug interactions by combining chemical, biological, and real-world clinical information. The project integrates data from public and curated drug databases, performs automated feature engineering, trains machine learning models, generates explainable predictions, and exposes the resulting inference pipeline through a Streamlit application.

The system is designed as an end-to-end ML pipeline rather than a standalone model, covering data acquisition, preprocessing, feature generation, model training, evaluation, explainability, and inference.

---

## Overview

Drug-drug interactions (DDIs) occur when one drug changes the effect, metabolism, or safety profile of another drug. Predicting these interactions computationally requires combining information from multiple sources because no single data modality completely describes the relationship between two drugs.

This project addresses the problem using a multi-modal approach that combines:

* Chemical structure information
* Biological and molecular target information
* Clinical adverse-event information
* Known drug-drug interaction information
* Mechanism-related information

The resulting pipeline is designed to support interaction prediction for individual drug pairs as well as medication combinations involving multiple concurrent drugs.

The project emphasizes reproducible data processing, modular Python components, model evaluation, explainability, and deployable inference.

---

## Key Objectives

The project has five primary objectives:

1. Build an automated pipeline for acquiring and processing drug-related data.
2. Extract complementary chemical, biological, and clinical features.
3. Train machine learning models for drug-drug interaction prediction.
4. Provide interpretable predictions using feature-level explanations and interaction mechanisms.
5. Deploy the prediction workflow through an interactive Streamlit application.

---

## System Architecture

```text
                         External Data Sources
                                 |
             +-------------------+-------------------+
             |                   |                   |
          PubChem            DrugBank             OpenFDA
             |                   |                   |
             +-------------------+-------------------+
                                 |
                                 v
                       Data Acquisition Layer
                                 |
                                 v
                       Data Cleaning & Validation
                                 |
                                 v
                    Multi-Modal Feature Engineering
                                 |
              +------------------+------------------+
              |                  |                  |
              v                  v                  v
       Molecular Features   Biological Features  Clinical Features
              |                  |                  |
              +------------------+------------------+
                                 |
                                 v
                          Feature Fusion
                                 |
                                 v
                       Machine Learning Models
                                 |
                                 v
                       Model Evaluation Layer
                                 |
                    +------------+------------+
                    |                         |
                    v                         v
             Prediction Engine         Explainability
                    |                  SHAP + Mechanisms
                    +------------+------------+
                                 |
                                 v
                          Streamlit Interface
```

---

## Project Structure

```text
drug-interaction-prediction/
│
├── config/
│   └── Configuration files and pipeline settings
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── docs/
│   └── team_responsibilities.md
│
├── models/
│   └── Trained model artifacts
│
├── notebooks/
│   └── Exploratory data analysis and experimentation
│
├── results/
│   ├── metrics/
│   ├── figures/
│   └── predictions/
│
├── scripts/
│   ├── download_data.py
│   ├── train_models.py
│   └── continuous_learning.py
│
├── src/
│   ├── data/
│   │   ├── pubchem_api.py
│   │   ├── drugbank_api.py
│   │   └── openfda_api.py
│   │
│   ├── features/
│   │   ├── molecular_features.py
│   │   ├── biological_features.py
│   │   ├── clinical_features.py
│   │   └── feature_fusion.py
│   │
│   ├── models/
│   │   └── baseline/
│   │
│   ├── explainability/
│   │   └── SHAP and mechanism analysis
│   │
│   └── inference/
│       └── Prediction and inference components
│
├── streamlit_app/
│   ├── app.py
│   └── pages/
│
├── .env.example
├── requirements.txt
└── README.md
```

---

## Data Pipeline

The data pipeline is designed to separate external data acquisition from feature engineering and model training.

### 1. Data Acquisition

The system collects information from multiple sources:

| Source   | Data Used                                              | Access             |
| -------- | ------------------------------------------------------ | ------------------ |
| PubChem  | SMILES, molecular formula and chemical information     | Public             |
| DrugBank | Drug information, targets, enzymes and DDI information | API/account access |
| OpenFDA  | FDA adverse-event reports                              | Public API         |

The API-specific functionality is implemented as separate Python modules under `src/data/`.

This separation allows individual data sources to be updated or replaced without changing the rest of the ML pipeline.

### 2. Data Processing

Retrieved data is processed before entering the model pipeline.

Processing includes:

* Schema normalization
* Missing-value handling
* Duplicate removal
* Identifier normalization
* Drug-name matching
* Data validation
* Cross-source integration
* Transformation into model-ready representations

The pipeline separates raw, intermediate, and processed data to maintain traceability between source data and model inputs.

### 3. Feature Engineering

The project uses multiple feature groups to represent drugs and their interactions.

#### Molecular Features

Chemical structures are represented using molecular descriptors and structure-derived information obtained from drug records.

Examples include:

* SMILES representations
* Molecular fingerprints
* Molecular descriptors
* Chemical similarity information

#### Biological Features

Biological information is derived from drug-target and drug-enzyme relationships.

These features capture relationships that may contribute to pharmacological interactions.

#### Clinical Features

Clinical information is derived from real-world adverse-event data.

This provides an additional signal beyond chemical and biological similarity by incorporating observations from real-world drug use.

### 4. Feature Fusion

The different feature groups are combined into a unified representation before model training.

```text
Molecular Features
        |
Biological Features -----> Feature Fusion -----> ML Model
        |
Clinical Features
```

This allows the model to use complementary information rather than relying on a single representation of a drug.

---

## Machine Learning Pipeline

The ML pipeline follows the following workflow:

```text
Processed Data
      |
      v
Feature Construction
      |
      v
Feature Validation
      |
      v
Train / Validation / Test Data
      |
      v
Model Training
      |
      v
Model Evaluation
      |
      v
Model Selection
      |
      v
Saved Model
      |
      v
Inference Engine
```

The model-training workflow is implemented independently from the Streamlit interface so that models can be trained and evaluated without starting the application.

This separation also allows the inference layer to consume previously trained model artifacts.

---

## Prediction

The inference layer accepts drug information and produces an interaction prediction.

The prediction workflow is:

```text
Input Drug(s)
     |
     v
Drug Identification
     |
     v
Feature Retrieval
     |
     v
Feature Construction
     |
     v
Feature Fusion
     |
     v
Trained ML Model
     |
     +----------------------+
     |                      |
     v                      v
Interaction Prediction   Explanation
     |                      |
     +----------+-----------+
                |
                v
        Streamlit Interface
```

The system is designed to support combinations of multiple concurrent medications rather than being restricted to isolated pairwise analysis.

---

## Explainability

Model predictions are complemented by explainability components to make the output easier to interpret.

### SHAP-Based Analysis

SHAP-based analysis is used to investigate the contribution of input features to model predictions.

This provides information about:

* Features contributing to a prediction
* Relative feature importance
* Direction of feature contributions
* Model-level feature importance

### Mechanism-Based Explanation

Where supporting information is available, the system associates predicted interactions with known biological or pharmacological mechanisms.

This provides contextual information alongside the numerical prediction rather than presenting the model output as an unexplained classification.

---

## Streamlit Application

The project includes an interactive Streamlit interface for model inference.

The application provides a workflow for:

1. Selecting or entering drugs
2. Retrieving relevant drug information
3. Generating model features
4. Running the trained prediction model
5. Displaying interaction predictions
6. Presenting explanatory information
7. Reviewing supporting molecular and clinical information

The UI is separated from the underlying inference implementation so that the prediction engine can also be reused independently of the web interface.

---

## Continuous Learning

The project includes a dedicated continuous-learning workflow through:

```text
scripts/continuous_learning.py
```

The intended workflow is:

```text
Updated External Data
        |
        v
Data Acquisition
        |
        v
Data Validation
        |
        v
Feature Generation
        |
        v
Model Retraining
        |
        v
Model Evaluation
        |
        v
Updated Model Artifact
```

This architecture allows newly available information to be incorporated into subsequent training cycles instead of requiring the complete project to be rebuilt manually.

Model updates should be evaluated before replacing an existing model artifact.

---

## Reproducibility

The project separates configuration, data acquisition, model training, inference, and presentation layers.

This allows individual stages of the pipeline to be executed independently.

The primary workflow is:

```bash
python scripts/download_data.py
python scripts/train_models.py
streamlit run streamlit_app/app.py
```

---

## Installation

### Requirements

The project requires:

* Python 3
* Git
* pip
* Internet connectivity for external API access

Clone the repository:

```bash
git clone https://github.com/tanishasnaik/Drug-Drug-Interaction.git
cd Drug-Drug-Interaction
```

Create a virtual environment:

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure environment variables using the provided environment configuration:

```bash
cp .env.example .env
```

On Windows, the `.env` file can be created manually from `.env.example`.

API credentials should be stored as environment variables rather than committed to the repository.

---

## Running the Pipeline

### Data Acquisition

```bash
python scripts/download_data.py
```

This executes the data acquisition stage and stores the retrieved information in the project data directories.

### Model Training

```bash
python scripts/train_models.py
```

This executes the model training workflow and stores the resulting model artifacts.

### Streamlit Application

```bash
streamlit run streamlit_app/app.py
```

The application can then be accessed through the local Streamlit server.

---

## Software Engineering Practices

The project is organized around modular components rather than a single monolithic training script.

Key engineering practices include:

* Modular Python components
* Separation of data acquisition and model training
* Configuration-driven execution
* Reusable feature-engineering modules
* Separate inference layer
* Environment-based API configuration
* Persisted model artifacts
* Reproducible training workflow
* Automated pipeline scripts
* Version-controlled source code

The architecture is intended to make individual components independently testable and maintainable.

---

## Testing

Automated testing can be applied to the individual pipeline components, including:

* Data ingestion
* API response processing
* Data validation
* Feature generation
* Model input dimensions
* Prediction output
* Inference behavior

Tests should be executed independently of the Streamlit UI so that data and model components can be validated automatically.

---

## Results and Evaluation

The project evaluates model behavior using classification and prediction metrics appropriate to the interaction-prediction task.

Evaluation outputs are stored under:

```text
results/
├── metrics/
├── figures/
└── predictions/
```

The evaluation workflow is separated from inference so that model performance can be measured on held-out data before deployment.

---

## Design Considerations

### Multi-Modal Data

Chemical, biological, and clinical data provide different representations of drug behavior. Combining these modalities allows the model pipeline to use information that would not be available from a single source.

### Data Quality

External data sources may contain missing, duplicated, inconsistent, or differently formatted records. The preprocessing layer therefore performs normalization and validation before features are passed to the model.

### Model Explainability

Prediction accuracy alone does not describe why a model produced a particular result. SHAP-based feature analysis and mechanism-related information are therefore incorporated into the prediction workflow.

### Deployment Separation

Training and inference are separated so that a trained model can be used without retraining whenever a prediction is requested.

---

## Limitations

The system has several practical limitations:

* External APIs may have rate limits or availability constraints.
* Drug databases can contain incomplete or differently structured information.
* Clinical adverse-event reports are observational and may contain reporting biases.
* Model performance depends on the quality and coverage of the available training data.
* Predictions should not be interpreted as clinical diagnoses or treatment recommendations.
* Continuous retraining requires appropriate validation before a newly trained model is deployed.

---

## Future Development

The modular architecture provides several possible directions for further development:

* Additional drug databases and data sources
* More advanced graph-based interaction models
* Improved molecular representations
* Automated data-quality monitoring
* Expanded automated test coverage
* Model versioning and experiment tracking
* API-based model serving
* Containerized deployment
* Automated CI/CD validation
* Monitoring of model performance after deployment

---

## Project Technologies

### Programming

* Python

### Machine Learning

* Scikit-learn
* Machine-learning classification models
* SHAP

### Data Processing

* NumPy
* Pandas

### Chemical / Biological Data

* PubChem
* DrugBank
* OpenFDA

### Application

* Streamlit

### Development

* Git
* GitHub
* Jupyter Notebook
* Python virtual environments

---

## Responsible Use

This project is intended for research and educational purposes.

Drug-interaction predictions generated by the system should not be used as a substitute for professional medical advice, diagnosis, or treatment decisions.

Clinical decisions should be made using validated clinical resources and by qualified healthcare professionals.

---

## License

This project is released under the MIT License.

---

## Authors

Developed as a collaborative machine-learning and software-engineering project.

The repository is structured into independent data, feature-engineering, modelling, explainability, inference, and application components to support collaborative development and maintainability.
