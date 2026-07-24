"""
ml_model.py — Load trained model and make predictions.
Combines ML predictions with clinical rules for robust assessments.
"""
import os
import pickle
import numpy as np

# Needed so pickle can resolve TorchMLPClassifier when the saved model is the
# neural net rather than XGBoost. Import path must match train_model.py's —
# see the comment there.
try:
    from backend.nn_model import TorchMLPClassifier  # noqa: F401
    from backend.drug_features import get_pharmacology_features, get_interaction_features
except ImportError:
    from nn_model import TorchMLPClassifier  # noqa: F401
    from drug_features import get_pharmacology_features, get_interaction_features

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class DrugRiskPredictor:
    """Loads the trained drug risk model and provides predictions."""

    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.feature_cols = None
        self.risk_labels = ["Low Risk", "Moderate", "High Risk", "Critical"]
        self.loaded = False
        self.active_model_name = None
        self.accuracy = None
        self.macro_f1 = None
        self.candidates = {}
        self._load_model()

    def _load_model(self):
        model_path = os.path.join(MODEL_DIR, "drug_risk_model.pkl")
        if not os.path.exists(model_path):
            print(f"[ML Model] No trained model found at {model_path}")
            return

        try:
            with open(model_path, "rb") as f:
                artifacts = pickle.load(f)
            self.model = artifacts["model"]
            self.label_encoder = artifacts["label_encoder"]
            self.feature_cols = artifacts["feature_cols"]
            self.risk_labels = artifacts.get("risk_labels", self.risk_labels)
            self.active_model_name = artifacts.get("active_model_name", "xgboost")
            self.accuracy = artifacts.get("accuracy")
            self.macro_f1 = artifacts.get("macro_f1")
            self.candidates = artifacts.get("candidates", {})
            self.loaded = True
            print(
                f"[ML Model] Loaded '{self.active_model_name}' "
                f"(accuracy: {self.accuracy}, macro F1: {self.macro_f1})"
            )
        except Exception as e:
            print(f"[ML Model] Error loading model: {e}")

    def predict(self, drug_name, patient_age=50, patient_sex="F",
                patient_weight=70, reaction_count=0, num_concomitant_drugs=0,
                known_side_effects=0, mean_se_freq=0, max_se_freq=0,
                current_meds=None):
        """
        Predict risk category for a drug-patient combination.
        Returns: dict with score, category, confidence, and label.

        current_meds: the patient's other medications (real drug names, e.g.
        from PatientProfile.current_meds), used to compute the same
        concomitant-drug interaction features (NSAID count, etc.) that
        training builds from FAERS' concomitant drug lists. Without this the
        model would be scored on features it never saw a real value for.
        """
        if not self.loaded:
            return {
                "score": None,
                "category": None,
                "label": "Model not available",
                "confidence": 0,
                "available": False,
            }

        # Encode drug name
        drug_lower = drug_name.lower().strip()
        # Handle paracetamol/acetaminophen alias
        if drug_lower == "paracetamol":
            drug_lower = "acetaminophen"

        try:
            drug_encoded = self.label_encoder.transform([drug_lower])[0]
        except ValueError:
            # Drug not in training data
            return {
                "score": None,
                "category": None,
                "label": f"Drug '{drug_name}' not in training data",
                "confidence": 0,
                "available": False,
            }

        # Encode sex
        sex_encoded = 1 if patient_sex.upper() in ("F", "FEMALE") else 0

        # Build feature vector matching the 11-feature schema
        # For new assessments, we use neutral defaults for reaction-based features
        # since the reactions haven't happened yet.
        weighted_reaction_score = 1.0  # Neutral baseline
        is_primary_suspect = 1        # Assessed as the target drug

        # Feature values by name, not position: self.feature_cols is
        # whatever list the loaded model was actually trained with (older
        # artifacts won't have the pharmacology/interaction columns at all),
        # so building a dict and selecting by name keeps this correct for
        # both old and new model files instead of hardcoding column order.
        available_features = {
            "drug_encoded": drug_encoded,
            "patient_age": patient_age,
            "sex_encoded": sex_encoded,
            "patient_weight": patient_weight,
            "reaction_count": reaction_count,
            "weighted_reaction_score": weighted_reaction_score,
            "is_primary_suspect": is_primary_suspect,
            "num_concomitant_drugs": num_concomitant_drugs,
            "known_side_effects_count": known_side_effects,
            "mean_se_frequency": mean_se_freq,
            "max_se_frequency": max_se_freq,
            **get_pharmacology_features(drug_lower),
            **get_interaction_features(current_meds),
        }

        try:
            feature_row = [available_features[col] for col in self.feature_cols]
        except KeyError as e:
            return {
                "score": None,
                "category": None,
                "label": f"Model expects unknown feature {e} — retrain needed",
                "confidence": 0,
                "available": False,
            }

        features = np.array([feature_row])

        # Predict
        try:
            pred_class = int(self.model.predict(features)[0])
            pred_proba = self.model.predict_proba(features)[0]
            confidence = float(max(pred_proba)) * 100

            return {
                "score": float(pred_class) / 3.0,  # Normalize to 0-1
                "category": pred_class,
                "label": self.risk_labels[pred_class],
                "confidence": round(confidence, 1),
                "probabilities": {
                    self.risk_labels[i]: round(float(p) * 100, 1)
                    for i, p in enumerate(pred_proba)
                    if i < len(self.risk_labels)
                },
                "available": True,
            }
        except Exception as e:
            return {
                "score": None,
                "category": None,
                "label": f"Prediction error: {str(e)}",
                "confidence": 0,
                "available": False,
            }


# Singleton instance
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = DrugRiskPredictor()
    return _predictor
