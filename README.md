# AI-Powered Dynamic Mental Health Monitoring and Distress Prediction System for Victims of Atrocities

> **This system is an academic screening and machine-learning prototype. It is not a medical diagnostic tool and should not replace professional mental-health assessment.**

## Run on Windows (Command Prompt)
Requires Python 3.9+ (tick "Add Python to PATH" when installing).
```bat
cd C:\path\to\mental_health_monitoring
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python tests\test_all.py
streamlit run app.py
```
Or just double-click `setup_and_run.bat`. The app opens at http://localhost:8501.
PowerShell: activate with `venv\Scripts\Activate.ps1` (if blocked, run `Set-ExecutionPolicy -Scope Process Bypass` first).

Optional: `python generate_dataset.py` then `python train_model.py` regenerate the data and model
(the app also retrains automatically if `models\model.joblib` is missing or incompatible).

## Modules
| File | Role |
|---|---|
| `generate_dataset.py` | 4,000-row **synthetic** dataset from hidden latent variables + noise + 4% label flips (seed 42) |
| `train_model.py` | Logistic Regression vs Random Forest; stratified split; scaler inside Pipeline (no leakage); selection by 5-fold CV macro-F1 on train only; test metrics reported |
| `predictor.py` | Loads model, class probabilities, risk score, local + global explanations |
| `risk_analysis.py` | Transparent 0–100 wellbeing score, indicator flags |
| `monitoring.py` | CSV history, linear-slope trend (Improving/Stable/Worsening), early-warning rules |
| `recommendations.py` | Rule-based wellbeing suggestions (no LLM) |
| `charts.py` / `app.py` | Matplotlib charts / Streamlit dashboard (7 pages) |

## Key definitions
- **Risk score** = 50·P(Moderate) + 100·P(High). **Wellbeing score** is a separate weighted composite (see `risk_analysis.py`).
- **Local explanation**: change in P(High) when an indicator is replaced by its training median.
- **Early warning**: latest session vs mean of previous ≤3 sessions; fires if ≥2 rules trigger (stress +2, anxiety +2, trauma +2, mood −2, sleep −1.5) or risk score jumps ≥15.
- Privacy: no names/identifiers; history in `data/monitoring_history.csv`.

## Limitations
Synthetic data means reported metrics do not reflect real-world performance. Explanations show model associations, not medical causes.
