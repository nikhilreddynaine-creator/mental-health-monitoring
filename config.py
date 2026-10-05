"""Shared constants for the monitoring platform."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "dataset.csv"
HISTORY_PATH = BASE_DIR / "data" / "monitoring_history.csv"
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
SEED = 42

# name -> (label, min, max, default, help text)
FEATURES = {
    "age": ("Age", 16, 80, 30, "Age in years."),
    "sleep_hours": ("Sleep Hours", 0.0, 12.0, 7.0, "Average hours slept per night recently."),
    "mood": ("Mood Score", 0.0, 10.0, 6.0, "0 = very low mood, 10 = very good mood."),
    "stress": ("Stress Level", 0.0, 10.0, 5.0, "0 = no stress, 10 = extreme stress."),
    "anxiety": ("Anxiety Level", 0.0, 10.0, 4.0, "0 = calm, 10 = constant nervousness/worry."),
    "social_interaction": ("Social Interaction", 0.0, 10.0, 5.0, "0 = none, 10 = very frequent, supportive contact."),
    "social_isolation": ("Social Isolation Indicator", 0.0, 10.0, 4.0, "0 = well connected, 10 = very isolated."),
    "activity_level": ("Daily Activity Level", 0.0, 10.0, 5.0, "0 = almost inactive, 10 = very active/engaged."),
    "trauma_symptoms": ("Trauma-related Symptoms", 0.0, 10.0, 3.0, "Overall burden of distressing trauma-related reactions."),
    "intrusive_thoughts": ("Intrusive Thoughts Frequency", 0.0, 10.0, 3.0, "How often unwanted distressing memories/thoughts occur."),
    "nightmares": ("Nightmares / Sleep Disturbance", 0.0, 10.0, 3.0, "Frequency of nightmares or disturbed sleep."),
    "hypervigilance": ("Hypervigilance Indicator", 0.0, 10.0, 3.0, "Feeling constantly on guard or easily startled."),
}
FEATURE_NAMES = list(FEATURES.keys())
CLASS_NAMES = ["Low", "Moderate", "High"]
DISCLAIMER = ("This system is an academic screening and machine-learning prototype. It is not a "
              "medical diagnostic tool and should not replace professional mental-health assessment.")
