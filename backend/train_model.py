"""
train_model.py — Train and compare drug risk prediction models (XGBoost + a
small PyTorch neural net) on merged FAERS + SIDER training data.
"""
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

# Import path must match how ml_model.py resolves it at unpickling time —
# both are run as `-m backend.<module>` from the repo root (see README /
# Dockerfile CMD), so `backend.nn_model` resolves first in both processes.
try:
    from backend.nn_model import TorchMLPClassifier, HAS_TORCH
    from backend.drug_features import PHARMACOLOGY_FEATURE_NAMES, INTERACTION_FEATURE_NAMES
except ImportError:
    from nn_model import TorchMLPClassifier, HAS_TORCH
    from drug_features import PHARMACOLOGY_FEATURE_NAMES, INTERACTION_FEATURE_NAMES

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def train():
    print("=" * 60)
    print("Training Optimized Drug Risk Prediction Model")
    print("=" * 60)

    # Load training data
    data_path = os.path.join(DATA_DIR, "training_data.csv")
    if not os.path.exists(data_path):
        print(f"Training data not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} training samples")

    # Encode drug names
    le_drug = LabelEncoder()
    df["drug_encoded"] = le_drug.fit_transform(df["drug_name"])

    # Optimized Feature columns
    feature_cols = [
        "drug_encoded",
        "patient_age",
        "sex_encoded",
        "patient_weight",
        "reaction_count",
        "weighted_reaction_score",
        "is_primary_suspect",
        "num_concomitant_drugs",
        "known_side_effects_count",
        "mean_se_frequency",
        "max_se_frequency",
        *PHARMACOLOGY_FEATURE_NAMES,
        *INTERACTION_FEATURE_NAMES,
    ]

    X = df[feature_cols].copy()
    y = df["risk_category"].copy()

    # Fill NaNs
    X = X.fillna(0)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    risk_labels = ["Low Risk", "Moderate", "High Risk", "Critical"]
    present_labels = sorted(y_test.unique())
    target_names = [risk_labels[i] for i in present_labels]

    candidates = {}  # name -> (model, accuracy, macro_f1)

    # ── XGBoost (tree-based baseline) ────────────────────────────────────────
    if HAS_XGBOOST:
        print("\nTuning XGBoost hyperparameters...")
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_weight': [1, 3, 5]
        }

        xgb_kwargs = {"random_state": 42, "eval_metric": "mlogloss", "use_label_encoder": False}
        if HAS_CUDA:
            print(f"[train_model] XGBoost training on GPU: {torch.cuda.get_device_name(0)}")
            xgb_kwargs["device"] = "cuda"
            xgb_kwargs["tree_method"] = "hist"  # required alongside device="cuda" for GPU histograms
        xgb = XGBClassifier(**xgb_kwargs)
        # n_jobs=1, not -1: RandomizedSearchCV's multiprocess worker spawn is
        # unreliable/slow on Windows and hung for 40+ minutes on this exact
        # dataset size during development. Each individual XGBoost fit is
        # already fast (more so on GPU), so CV-fold parallelism isn't needed.
        #
        # scoring='f1_macro', not 'accuracy': risk_category is imbalanced
        # (Low >> Critical, see class weighting below), so a search that
        # optimizes accuracy will happily pick hyperparameters that just
        # predict the majority class more often — the exact trap this
        # project's final model-selection step (macro F1) was already meant
        # to avoid, except the search itself was still using the trap metric.
        random_search = RandomizedSearchCV(
            xgb, param_distributions=param_dist, n_iter=10,
            scoring='f1_macro', n_jobs=1, cv=3, random_state=42, verbose=1
        )
        # Class-weighted samples: same rationale as the NN's class_weight
        # below, applied via XGBoost's sample_weight instead since it isn't
        # a torch model with its own loss-function weight argument.
        train_class_counts = y_train.value_counts()
        class_to_weight = (train_class_counts.sum() / train_class_counts).to_dict()
        xgb_sample_weight = y_train.map(class_to_weight).values

        random_search.fit(X_train, y_train, sample_weight=xgb_sample_weight)
        xgb_model = random_search.best_estimator_
        if HAS_CUDA:
            # Saved/served model must load on CPU-only deployments (e.g. a
            # GPU-less Hugging Face Space) — inference for a model this size
            # is sub-millisecond on CPU regardless, so there's no cost to this.
            xgb_model.set_params(device="cpu")
        print(f"Best params: {random_search.best_params_}")
    else:
        print("\nXGBoost not found, using baseline GradientBoosting...")
        xgb_model = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
        xgb_model.fit(X_train, y_train)

    xgb_pred = xgb_model.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_f1 = f1_score(y_test, xgb_pred, average="macro", labels=present_labels, zero_division=0)
    candidates["xgboost"] = (xgb_model, xgb_acc, xgb_f1)
    print(f"\n[XGBoost] Test Accuracy: {xgb_acc:.4f}  |  Macro F1: {xgb_f1:.4f}")
    print(classification_report(y_test, xgb_pred, target_names=target_names, labels=present_labels))

    # ── Neural network (gradient-based, on standardized features) ───────────
    if HAS_TORCH:
        print("\nTraining neural network (PyTorch MLP)...")
        # FAERS risk categories are heavily skewed toward "Low" (most adverse
        # event reports aren't fatal/life-threatening) — inverse-frequency
        # class weights keep the NN from collapsing to the majority class.
        class_counts = y_train.value_counts().reindex(range(len(risk_labels)), fill_value=0)
        class_weight = (class_counts.sum() / (class_counts + 1)).values
        class_weight = (class_weight / class_weight.sum() * len(risk_labels)).tolist()

        nn_model = TorchMLPClassifier(
            input_dim=len(feature_cols),
            num_classes=len(risk_labels),
            hidden=(64, 32),
            epochs=120,
            lr=1e-3,
            class_weight=class_weight,
        )
        nn_model.fit(X_train.values, y_train.values)

        nn_pred = nn_model.predict(X_test.values)
        nn_acc = accuracy_score(y_test, nn_pred)
        nn_f1 = f1_score(y_test, nn_pred, average="macro", labels=present_labels, zero_division=0)
        candidates["neural_net"] = (nn_model, nn_acc, nn_f1)
        print(f"\n[Neural Net] Test Accuracy: {nn_acc:.4f}  |  Macro F1: {nn_f1:.4f}")
        print(classification_report(y_test, nn_pred, target_names=target_names, labels=present_labels))
    else:
        print("\nPyTorch not found — skipping neural network training (pip install torch).")

    # ── Pick the winner ───────────────────────────────────────────────────────
    # Macro F1 is the selection metric, not accuracy: risk_category is
    # imbalanced (Low >> Critical), so a model can post a high accuracy just
    # by always predicting "Low" while being useless for the rare, high-stakes
    # classes this tool actually exists to flag.
    best_name = max(candidates, key=lambda k: candidates[k][2])
    best_model, best_acc, best_f1 = candidates[best_name]

    print(f"\n{'=' * 60}")
    print("Model comparison (selection metric: macro F1)")
    for name, (_, acc, f1) in candidates.items():
        marker = " <- selected" if name == best_name else ""
        print(f"  {name:12s}  accuracy={acc:.4f}  macro_f1={f1:.4f}{marker}")
    print(f"{'=' * 60}")

    # Save model artifacts. "model" stays the generic key ml_model.py already
    # reads, set to whichever candidate won; the rest are kept for comparison
    # and so a future run can prefer/ensemble across both.
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "drug_risk_model.pkl")

    artifacts = {
        "model": best_model,
        "active_model_name": best_name,
        "label_encoder": le_drug,
        "feature_cols": feature_cols,
        "risk_labels": risk_labels,
        "accuracy": best_acc,
        "macro_f1": best_f1,
        "candidates": {
            name: {"accuracy": acc, "macro_f1": f1}
            for name, (_, acc, f1) in candidates.items()
        },
    }

    with open(model_path, "wb") as f:
        pickle.dump(artifacts, f)

    print(f"\nModel saved to: {model_path} (active model: {best_name})")
    return best_model, le_drug


if __name__ == "__main__":
    train()
