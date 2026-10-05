"""Loads the saved pipeline and produces predictions + model-based explanations."""
import joblib
import numpy as np
import pandas as pd
from config import MODEL_PATH, FEATURE_NAMES, CLASS_NAMES


class ModelNotFoundError(FileNotFoundError):
    pass


def load_bundle(path=MODEL_PATH):
    if not path.exists():
        raise ModelNotFoundError(f"Model file not found: {path}. Run `python train_model.py` first.")
    return joblib.load(path)


def predict(bundle, inputs: dict):
    """Return class probabilities, category and a 0-100 expected-risk score."""
    X = pd.DataFrame([{f: float(inputs[f]) for f in FEATURE_NAMES}])
    proba = bundle["pipeline"].predict_proba(X)[0]
    probs = dict(zip(CLASS_NAMES, [float(p) for p in proba]))
    category = CLASS_NAMES[int(np.argmax(proba))]
    risk_score = 50 * probs["Moderate"] + 100 * probs["High"]   # expected severity, 0-100
    return dict(category=category, probabilities=probs, risk_score=float(risk_score))


def local_contributions(bundle, inputs: dict):
    """Model-agnostic local explanation: change in P(High) when one indicator is
    replaced by its training-set median. Positive = this value pushes High-risk probability up
    relative to a typical value. Works for both Logistic Regression and Random Forest."""
    pipe, med = bundle["pipeline"], bundle["medians"]
    base = {f: float(inputs[f]) for f in FEATURE_NAMES}
    rows = [base] + [{**base, f: med[f]} for f in FEATURE_NAMES]
    p_high = pipe.predict_proba(pd.DataFrame(rows))[:, 2]
    return {f: float(p_high[0] - p_high[i + 1]) for i, f in enumerate(FEATURE_NAMES)}


def global_importance(bundle):
    """Importance of the *selected* model (RF feature_importances_ or mean |LR coefficient|)."""
    return bundle["importances"][bundle["model_name"]]


def load_or_train(path=MODEL_PATH):
    """Load the saved model; if missing or unreadable (e.g. different scikit-learn version), retrain locally."""
    try:
        return load_bundle(path)
    except Exception:
        from train_model import train
        train(model_path=path)
        return load_bundle(path)
