"""
main.py
--------
Single entry point for the entire Drug Interaction Prediction pipeline.

Run this ONE file to do everything:
    python main.py

It will show a menu and let you choose what to run.
"""

import os
import sys
import subprocess

# ── Make sure we can import from src/ ───────────────────────────
sys.path.insert(0, os.path.abspath("."))


# ── Colors for terminal output ───────────────────────────────────
class C:
    HEADER  = "\033[95m"
    BLUE    = "\033[94m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    BOLD    = "\033[1m"
    END     = "\033[0m"


def banner():
    print(f"""
{C.CYAN}{C.BOLD}
╔══════════════════════════════════════════════════════════════╗
║       💊  Drug Interaction Prediction System                 ║
║       Multi-Modal Explainable AI                             ║
╚══════════════════════════════════════════════════════════════╝
{C.END}""")


def divider():
    print(f"{C.BLUE}{'─' * 62}{C.END}")


def success(msg):
    print(f"{C.GREEN}  ✅  {msg}{C.END}")


def info(msg):
    print(f"{C.CYAN}  ℹ️   {msg}{C.END}")


def warn(msg):
    print(f"{C.YELLOW}  ⚠️   {msg}{C.END}")


def error(msg):
    print(f"{C.RED}  ❌  {msg}{C.END}")


def step_header(num, title):
    print(f"\n{C.BOLD}{C.BLUE}[ STEP {num} ] {title}{C.END}")
    divider()


# ════════════════════════════════════════════════════════════════
# STEP 1 — Check environment
# ════════════════════════════════════════════════════════════════

def check_environment():
    step_header(1, "Checking Environment")

    # Check .env file
    if not os.path.exists(".env"):
        warn(".env file not found. Copying from .env.example...")
        if os.path.exists(".env.example"):
            import shutil
            shutil.copy(".env.example", ".env")
            success(".env created. You can add your API keys to it later.")
        else:
            error(".env.example not found. Something is wrong with the project files.")
            return False
    else:
        success(".env file found")

    # Check required packages
    required = ["pandas", "numpy", "sklearn", "rdkit", "pubchempy",
                "xgboost", "streamlit", "shap", "requests", "yaml", "loguru"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg if pkg != "yaml" else "yaml")
        except ImportError:
            missing.append(pkg)

    if missing:
        error(f"Missing packages: {', '.join(missing)}")
        info("Run:  pip install -r requirements.txt")
        return False
    else:
        success("All required packages are installed")

    # Check data directories exist
    for folder in ["data/raw/pubchem", "data/raw/drugbank", "data/raw/openfda",
                   "data/processed", "models/baseline", "results/metrics", "results/figures"]:
        os.makedirs(folder, exist_ok=True)
    success("Data directories verified")

    return True


# ════════════════════════════════════════════════════════════════
# STEP 2 — Download data from APIs
# ════════════════════════════════════════════════════════════════

def run_data_download():
    step_header(2, "Downloading Data from APIs")
    info("This fetches drug data from PubChem, DrugBank, and OpenFDA...")
    info("Estimated time: 3–8 minutes depending on your internet speed")
    print()

    try:
        from src.data.data_loader import load_all_data, build_pairs_dataset

        # Run the full ingestion
        data = load_all_data()

        if data["chemical"].empty:
            error("Failed to fetch chemical data from PubChem.")
            return False

        success(f"Chemical data: {len(data['chemical'])} drugs fetched")
        success(f"Biological data: {len(data['biological'])} drugs processed")
        success(f"Clinical data: {len(data['clinical'])} drugs processed")

        # Build the labeled pairs dataset
        df_pairs = build_pairs_dataset(
            data["chemical"],
            data["biological"],
            data["clinical"]
        )

        success(f"Drug pairs dataset: {len(df_pairs)} pairs saved to data/processed/interaction_pairs.csv")
        interaction_count = int(df_pairs["label"].sum())
        info(f"Interactions found: {interaction_count} / {len(df_pairs)} pairs")
        return True

    except Exception as e:
        error(f"Data download failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ════════════════════════════════════════════════════════════════
# STEP 3 — Build features
# ════════════════════════════════════════════════════════════════

def run_feature_engineering():
    step_header(3, "Building Multi-Modal Feature Matrix")
    info("Generating molecular fingerprints, enzyme vectors, and FDA features...")

    try:
        import pandas as pd
        from src.features.feature_fusion import build_full_feature_matrix
        import numpy as np

        df_pairs = pd.read_csv("data/processed/interaction_pairs.csv")
        df_bio   = pd.read_csv("data/raw/drugbank/enzyme_features.csv")
        df_clin  = pd.read_csv("data/raw/openfda/report_counts.csv")

        X, y, feature_names = build_full_feature_matrix(df_pairs, df_bio, df_clin)

        # Save for training step
        np.save("data/interim/X_features.npy", X)
        np.save("data/interim/y_labels.npy", y)

        success(f"Feature matrix shape: {X.shape}")
        success(f"  Chemical features:   {2048 * 2}")
        success(f"  Biological features: 40")
        success(f"  Clinical features:   3")
        success(f"  Total:               {X.shape[1]}")
        success(f"Features saved to data/interim/")
        return True, X, y, feature_names

    except FileNotFoundError:
        error("Processed data not found. Run Step 2 first.")
        return False, None, None, None
    except Exception as e:
        error(f"Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None


# ════════════════════════════════════════════════════════════════
# STEP 4 — Train models
# ════════════════════════════════════════════════════════════════

def run_model_training(X=None, y=None, feature_names=None):
    step_header(4, "Training Machine Learning Models")
    info("Training: Logistic Regression, Random Forest, XGBoost")

    try:
        import numpy as np

        # Load features if not passed in
        if X is None:
            try:
                X = np.load("data/interim/X_features.npy")
                y = np.load("data/interim/y_labels.npy")
                import json
                with open("models/metadata/feature_names.json") as f:
                    feature_names = json.load(f)
            except FileNotFoundError:
                error("Feature files not found. Run Step 3 first.")
                return False

        from src.models.model_trainer import ModelTrainer

        trainer = ModelTrainer()
        results = trainer.train_all(X, y, feature_names)

        print()
        for model_name, metrics in results.items():
            success(f"{model_name:<25} AUROC: {metrics['auroc']}  F1: {metrics['f1']}")

        success("All models saved to models/baseline/")
        success("Metrics saved to results/metrics/baseline_metrics.json")
        return True

    except Exception as e:
        error(f"Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ════════════════════════════════════════════════════════════════
# STEP 5 — Launch Streamlit app
# ════════════════════════════════════════════════════════════════

def run_app():
    step_header(5, "Launching Streamlit App")
    success("Starting the web app...")
    info("The app will open in your browser at: http://localhost:8501")
    info("Press Ctrl+C in the terminal to stop the app")
    print()

    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "streamlit_app/app.py"],
            check=True
        )
    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}App stopped.{C.END}")
    except FileNotFoundError:
        error("streamlit_app/app.py not found.")
    except subprocess.CalledProcessError as e:
        error(f"App failed to start: {e}")


# ════════════════════════════════════════════════════════════════
# QUICK PREDICT — test a drug pair without the app
# ════════════════════════════════════════════════════════════════

def run_quick_predict():
    step_header("Q", "Quick Drug Pair Prediction (Console)")
    print()

    drug_a = input(f"  {C.BOLD}Enter Drug A:{C.END} ").strip().lower()
    drug_b = input(f"  {C.BOLD}Enter Drug B:{C.END} ").strip().lower()

    if not drug_a or not drug_b:
        warn("Please enter both drug names.")
        return

    info(f"Analyzing: {drug_a} + {drug_b} ...")
    print()

    try:
        from src.inference.predictor import DDIPredictor
        predictor = DDIPredictor()
        result = predictor.predict_pair(drug_a, drug_b)

        if "error" in result:
            error(result["error"])
            return

        sev = result["severity"]
        mech = result["mechanism"]

        divider()
        print(f"\n  {C.BOLD}Drug Pair:{C.END}  {drug_a.capitalize()} + {drug_b.capitalize()}")
        print(f"  {C.BOLD}Risk Score:{C.END} {result['probability']:.1%}")
        print(f"  {C.BOLD}Severity:{C.END}  {sev['emoji']}  {sev['severity_label']} (Level {sev['severity_level']}/5)")
        print(f"  {C.BOLD}Recommendation:{C.END} {sev['recommendation']}")
        print()
        print(f"  {C.BOLD}Mechanism:{C.END}")
        print(f"  {mech['explanation']}")

        shared = mech.get("shared_enzymes", [])
        if shared:
            print(f"\n  {C.BOLD}Shared Enzymes:{C.END} {', '.join(shared)}")

        print(f"\n  {C.BOLD}FDA Adverse Reports:{C.END}")
        print(f"    {drug_a.capitalize()}: {result['fda_reports_a']} reports")
        print(f"    {drug_b.capitalize()}: {result['fda_reports_b']} reports")

        if result.get("demo_mode"):
            print()
            warn("Running in DEMO mode — train the model for real predictions.")
        divider()

    except Exception as e:
        error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()


# ════════════════════════════════════════════════════════════════
# FULL PIPELINE — run everything at once
# ════════════════════════════════════════════════════════════════

def run_full_pipeline():
    step_header("★", "Running Full Pipeline (Steps 1 → 4)")
    info("This will: check env → download data → build features → train models")
    warn("Estimated total time: 10–20 minutes")
    print()

    confirm = input(f"  {C.BOLD}Continue? (y/n):{C.END} ").strip().lower()
    if confirm != "y":
        info("Cancelled.")
        return

    # Step 1
    if not check_environment():
        error("Environment check failed. Fix issues above and try again.")
        return

    # Step 2
    if not run_data_download():
        error("Data download failed. Check your internet connection and try again.")
        return

    # Step 3
    ok, X, y, feature_names = run_feature_engineering()
    if not ok:
        error("Feature engineering failed.")
        return

    # Step 4
    if not run_model_training(X, y, feature_names):
        error("Model training failed.")
        return

    print()
    divider()
    success("Full pipeline complete!")
    success("Your models are trained and ready.")
    info("Run the app with option 5, or quick predict with option Q.")
    divider()


# ════════════════════════════════════════════════════════════════
# MAIN MENU
# ════════════════════════════════════════════════════════════════

def main_menu():
    banner()

    options = {
        "1": ("Check environment & setup",          check_environment),
        "2": ("Download data from APIs",             run_data_download),
        "3": ("Build feature matrix",                run_feature_engineering),
        "4": ("Train ML models",                     run_model_training),
        "5": ("Launch Streamlit app",                run_app),
        "Q": ("Quick predict (console)",             run_quick_predict),
        "A": ("Run FULL pipeline (steps 1→4)",       run_full_pipeline),
        "X": ("Exit",                                None),
    }

    while True:
        print(f"\n{C.BOLD}  What would you like to do?{C.END}\n")
        for key, (label, _) in options.items():
            prefix = f"{C.CYAN}[{key}]{C.END}"
            print(f"    {prefix}  {label}")

        print()
        choice = input(f"  {C.BOLD}Enter choice:{C.END} ").strip().upper()

        if choice == "X":
            print(f"\n{C.CYAN}  Goodbye! 👋{C.END}\n")
            break
        elif choice in options:
            _, fn = options[choice]
            if fn:
                fn()
        else:
            warn("Invalid choice. Enter a number 1–5, Q, A, or X.")


# ── Entry point ──────────────────────────────────────────────────
if __name__ == "__main__":

    # If a command-line argument is given, skip the menu
    # e.g.  python main.py --full   runs the full pipeline
    #        python main.py --app    launches the app directly
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "--full":
            banner()
            run_full_pipeline()
        elif arg == "--app":
            banner()
            run_app()
        elif arg == "--predict":
            banner()
            run_quick_predict()
        elif arg == "--data":
            banner()
            check_environment()
            run_data_download()
        elif arg == "--train":
            banner()
            ok, X, y, names = run_feature_engineering()
            if ok:
                run_model_training(X, y, names)
        else:
            print(f"Unknown argument: {arg}")
            print("Usage: python main.py [--full | --app | --predict | --data | --train]")
    else:
        main_menu()
