"""Synthetic dataset generator (fixed seed, reproducible).

Design: hidden latent variables (distress state, trauma exposure, resilience) drive
the *observed* noisy indicators. The label is derived from the hidden variables plus
noise and 4% label flips, so the classifier cannot trivially recover a formula and
classes overlap realistically. THE DATA IS SYNTHETIC - not real patient data.
"""
import numpy as np
import pandas as pd
from config import DATASET_PATH, SEED, CLASS_NAMES


def generate(n=4000, seed=SEED):
    rng = np.random.default_rng(seed)
    state = rng.normal(0, 1, n)           # hidden distress state
    exposure = rng.beta(2, 3, n)          # hidden trauma exposure severity
    resil = rng.normal(0, 1, n)           # hidden resilience
    N = lambda s: rng.normal(0, s, n)
    clip = lambda x, lo=0, hi=10: np.clip(np.round(x, 1), lo, hi)

    stress = clip(4.8 + 1.4 * state + 2.0 * (exposure - 0.4) + N(1.3))
    anxiety = clip(4.6 + 1.3 * state + 0.15 * (stress - 5) + N(1.4))
    mood = clip(5.8 - 1.2 * state + 0.4 * resil + N(1.5))
    sleep = clip(7.0 - 0.45 * state - 0.06 * (stress - 5) + N(1.0), 2, 11)
    social = clip(5.5 - 0.8 * state + 0.7 * resil + N(1.6))
    isolation = clip(7.0 - 0.55 * social + 0.5 * state + N(1.5))
    activity = clip(5.2 - 0.6 * state + 0.5 * resil + N(1.6))
    trauma = clip(3.5 + 1.2 * state + 3.0 * (exposure - 0.4) + N(1.5))
    intrusive = clip(1.0 + 0.6 * trauma + 0.6 * state + N(1.4))
    nightmares = clip(0.8 + 0.5 * trauma + 0.5 * state + N(1.6))
    hyper = clip(0.8 + 0.5 * trauma + 0.1 * (anxiety - 5) + 0.4 * state + N(1.5))
    age = np.clip(np.round(rng.normal(34, 12, n)), 16, 80).astype(int)

    # hidden risk (not observable) with an interaction term and noise
    lat = state + 3.0 * (exposure - 0.4) - 0.4 * resil + 0.35 * state * (exposure * 3) + N(0.6)
    lo, hi = np.quantile(lat, [0.40, 0.78])
    y = np.where(lat < lo, 0, np.where(lat < hi, 1, 2))
    flip = rng.random(n) < 0.04
    shift = rng.choice([-1, 1], n)
    y = np.where(flip, np.clip(y + shift, 0, 2), y)

    df = pd.DataFrame(dict(age=age, sleep_hours=sleep, mood=mood, stress=stress, anxiety=anxiety,
                           social_interaction=social, social_isolation=isolation, activity_level=activity,
                           trauma_symptoms=trauma, intrusive_thoughts=intrusive, nightmares=nightmares,
                           hypervigilance=hyper))
    df["risk_level"] = y
    return df


def main():
    df = generate()
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)
    print(f"Saved {len(df)} rows -> {DATASET_PATH}")
    print(df["risk_level"].map(dict(enumerate(CLASS_NAMES))).value_counts())


if __name__ == "__main__":
    main()
