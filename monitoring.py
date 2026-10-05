"""Local CSV monitoring history, trend calculation and early-warning rules."""
import numpy as np
import pandas as pd
from datetime import datetime
from config import HISTORY_PATH, FEATURE_NAMES

COLUMNS = ["timestamp"] + FEATURE_NAMES + ["wellbeing_score", "risk_score", "predicted_risk",
                                            "risk_probability", "prob_low", "prob_moderate", "prob_high"]


def load_history(path=HISTORY_PATH):
    """Returns an empty frame if the file is missing/empty/corrupt."""
    try:
        df = pd.read_csv(path)
        if not set(COLUMNS).issubset(df.columns):
            return pd.DataFrame(columns=COLUMNS)
        return df[COLUMNS].reset_index(drop=True)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=COLUMNS)


def add_record(inputs, wellbeing, pred, path=HISTORY_PATH, timestamp=None):
    p = pred["probabilities"]
    row = {"timestamp": timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           **{f: inputs[f] for f in FEATURE_NAMES}, "wellbeing_score": wellbeing,
           "risk_score": round(pred["risk_score"], 1), "predicted_risk": pred["category"],
           "risk_probability": round(p["High"], 4), "prob_low": round(p["Low"], 4),
           "prob_moderate": round(p["Moderate"], 4), "prob_high": round(p["High"], 4)}
    df = pd.concat([load_history(path), pd.DataFrame([row])], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def clear_history(path=HISTORY_PATH):
    pd.DataFrame(columns=COLUMNS).to_csv(path, index=False)


def _slope(y):
    return float(np.polyfit(np.arange(len(y)), np.asarray(y, float), 1)[0]) if len(y) >= 2 else 0.0


def compute_trend(df, window=5, risk_thr=3.0, wb_thr=2.0):
    """Linear-regression slope over last `window` sessions of risk_score and wellbeing_score (points/session)."""
    if len(df) < 2:
        return dict(status="Insufficient data", message="Record at least 2 sessions to see a trend.",
                    risk_slope=0.0, wellbeing_slope=0.0, n=len(df))
    recent = df.tail(window)
    rs, ws = _slope(recent["risk_score"]), _slope(recent["wellbeing_score"])
    n = len(recent)
    if rs >= risk_thr or ws <= -wb_thr:
        status, msg = "Worsening", "Recent screening results show increasing distress indicators."
    elif rs <= -risk_thr or ws >= wb_thr:
        status, msg = "Improving", f"Your wellbeing indicators have improved over the last {n} recorded sessions."
    else:
        status, msg = "Stable", f"Indicators have remained broadly stable over the last {n} recorded sessions."
    return dict(status=status, message=msg, risk_slope=rs, wellbeing_slope=ws, n=n)


def early_warning(df, drops=dict(stress=2.0, anxiety=2.0, trauma_symptoms=2.0, mood=-2.0, sleep_hours=-1.5)):
    """Compare the latest session with the mean of up to 3 previous sessions.
    Warning triggers when >= 2 rules fire, or when the risk score jumped >= 15 points."""
    if len(df) < 2:
        return dict(active=False, triggers=[], message="")
    last = df.iloc[-1]
    prev = df.iloc[max(0, len(df) - 4):-1]
    names = dict(stress="rapid increase in stress", anxiety="increasing anxiety",
                 trauma_symptoms="increasing trauma-related symptoms", mood="decreasing mood",
                 sleep_hours="rapid decrease in sleep")
    triggers = []
    for f, thr in drops.items():
        d = float(last[f] - prev[f].mean())
        if (thr > 0 and d >= thr) or (thr < 0 and d <= thr):
            triggers.append(f"{names[f]} ({d:+.1f})")
    jump = float(last["risk_score"] - prev["risk_score"].mean())
    if jump >= 15:
        triggers.append(f"risk score rose by {jump:.0f} points")
    active = len(triggers) >= 2 or jump >= 15
    msg = ("Recent wellbeing indicators show increased distress signals. "
           "Consider seeking support from a qualified professional.") if active else ""
    return dict(active=active, triggers=triggers, message=msg)
