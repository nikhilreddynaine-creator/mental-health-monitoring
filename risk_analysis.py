"""Wellbeing score, indicator flags and category helpers (transparent, rule-based)."""
from config import FEATURE_NAMES

# Weights sum to 1. Positive items raise the score, negative items lower it.
POSITIVE = {"sleep_hours": 0.12, "mood": 0.14, "social_interaction": 0.10, "activity_level": 0.06}
NEGATIVE = {"stress": 0.14, "anxiety": 0.12, "trauma_symptoms": 0.10, "intrusive_thoughts": 0.06,
            "nightmares": 0.05, "hypervigilance": 0.05, "social_isolation": 0.06}


def _sleep_quality(h):
    """1.0 at 7-9 h, falling linearly to 0 at 3 h or 12 h."""
    if 7 <= h <= 9:
        return 1.0
    return max(0.0, (h - 3) / 4) if h < 7 else max(0.0, 1 - (h - 9) / 3)


def wellbeing_score(inputs: dict) -> float:
    """Application-level composite 0-100 (NOT the ML prediction)."""
    s = 0.0
    for f, w in POSITIVE.items():
        v = _sleep_quality(inputs[f]) if f == "sleep_hours" else inputs[f] / 10
        s += w * min(max(v, 0), 1)
    for f, w in NEGATIVE.items():
        s += w * (1 - min(max(inputs[f] / 10, 0), 1))
    return round(100 * s, 1)


def score_label(score):
    return "Good" if score >= 65 else "Fair" if score >= 45 else "Low"


def detect_indicators(inputs: dict):
    """Flag indicators outside a reasonable range -> list of (feature, message, severity 1-3)."""
    out = []
    rules = [
        ("stress", lambda v: v >= 7, "High stress level"),
        ("anxiety", lambda v: v >= 7, "Elevated anxiety"),
        ("sleep_hours", lambda v: v < 5.5, "Low sleep duration"),
        ("mood", lambda v: v <= 3.5, "Low mood"),
        ("social_interaction", lambda v: v <= 3, "Low social interaction"),
        ("social_isolation", lambda v: v >= 7, "High social isolation"),
        ("trauma_symptoms", lambda v: v >= 7, "High trauma-related symptoms"),
        ("intrusive_thoughts", lambda v: v >= 7, "Frequent intrusive thoughts"),
        ("nightmares", lambda v: v >= 7, "Frequent nightmares / sleep disturbance"),
        ("hypervigilance", lambda v: v >= 7, "High hypervigilance"),
        ("activity_level", lambda v: v <= 2.5, "Very low daily activity"),
    ]
    for f, cond, msg in rules:
        v = inputs[f]
        if cond(v):
            ext = abs(v - 5) if f != "sleep_hours" else abs(v - 7)
            out.append((f, msg, 3 if ext >= 4 else 2 if ext >= 3 else 1))
    return sorted(out, key=lambda t: -t[2])


def risk_color(cat):
    return {"Low": "#2e9e6b", "Moderate": "#e0a030", "High": "#d9534f"}.get(cat, "#888")


def monitoring_status(category, trend_status=None):
    if category == "High" or trend_status == "Worsening":
        return "Needs Attention"
    if category == "Moderate":
        return "Monitor Closely"
    return "Stable"


def radar_values(inputs: dict):
    """0-10 'concern' scale where higher = more concern, for dashboard radar."""
    return {
        "Stress": inputs["stress"], "Anxiety": inputs["anxiety"], "Low Mood": 10 - inputs["mood"],
        "Poor Sleep": 10 * (1 - _sleep_quality(inputs["sleep_hours"])),
        "Low Social Connection": 10 - inputs["social_interaction"],
        "Trauma Symptoms": inputs["trauma_symptoms"],
    }
