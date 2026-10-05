"""AI-Powered Dynamic Mental Health Monitoring & Distress Prediction System (academic prototype)."""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import charts
import monitoring as mon
import risk_analysis as ra
from config import CLASS_NAMES, DISCLAIMER, FEATURES, FEATURE_NAMES, HISTORY_PATH, MODEL_PATH
from predictor import ModelNotFoundError, global_importance, load_or_train, local_contributions, predict
from recommendations import suggestions

st.set_page_config(page_title="AI Mental Health Monitoring", page_icon="🧠", layout="wide")
LABELS = {k: v[0] for k, v in FEATURES.items()}

CSS = """
<style>
.block-container {padding-top: 1.5rem; max-width: 1250px;}
.hero {background: linear-gradient(120deg,#1f2a44,#3b6fd4); color:white; padding:22px 28px; border-radius:16px; margin-bottom:18px;}
.hero h1 {margin:0; font-size:1.8rem; color:white;} .hero p {margin:4px 0 0; opacity:.85;}
.card {background:#fff; border:1px solid #e6eaf1; border-radius:14px; padding:16px 18px; box-shadow:0 1px 4px rgba(31,42,68,.06);}
.card .lab {font-size:.72rem; letter-spacing:.08em; color:#7a869a; font-weight:600;}
.card .val {font-size:1.9rem; font-weight:700; line-height:1.25;} .card .sub {font-size:.8rem; color:#7a869a;}
.warn {background:#fff4e5; border-left:5px solid #e0a030; padding:12px 16px; border-radius:8px; margin:10px 0;}
.disc {font-size:.78rem; color:#7a869a; border-top:1px solid #e6eaf1; padding-top:8px; margin-top:24px;}
.pill {display:inline-block; padding:4px 12px; border-radius:20px; background:#eef2fb; color:#1f2a44; margin:3px 4px 3px 0; font-size:.85rem;}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)


def card(label, value, sub="", color="#1f2a44"):
    st.markdown(f'<div class="card"><div class="lab">{label}</div>'
                f'<div class="val" style="color:{color}">{value}</div><div class="sub">{sub}</div></div>',
                unsafe_allow_html=True)


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def hero(title, sub):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


def disclaimer():
    st.markdown(f'<div class="disc">⚕️ {DISCLAIMER}</div>', unsafe_allow_html=True)


@st.cache_resource
def get_bundle():
    return load_or_train()


def analyze(bundle, inputs):
    pred = predict(bundle, inputs)
    return dict(inputs=inputs, pred=pred, wellbeing=ra.wellbeing_score(inputs),
                contrib=local_contributions(bundle, inputs))


def current(bundle):
    """Latest analysis this session, else rebuilt from the most recent stored record."""
    if st.session_state.get("current"):
        return st.session_state["current"]
    h = mon.load_history()
    if h.empty:
        return None
    return analyze(bundle, {f: float(h.iloc[-1][f]) for f in FEATURE_NAMES})


def need_data():
    st.info("No screening data yet. Open **🧠 Mental Health Screening** and click *Analyze Wellbeing*.")


# ---------------------------------------------------------------- pages
def page_dashboard(bundle):
    hero("AI Mental Health Monitoring", "Distress Prediction & Wellbeing Analysis")
    cur = current(bundle)
    if cur is None:
        need_data(); disclaimer(); return
    hist = mon.load_history(); trend = mon.compute_trend(hist)
    cat, p = cur["pred"]["category"], cur["pred"]["probabilities"]
    status = ra.monitoring_status(cat, trend["status"])
    c = st.columns(4)
    with c[0]: card("CURRENT RISK", cat.upper(), "Screening category", ra.risk_color(cat))
    with c[1]: card("WELLBEING SCORE", f'{cur["wellbeing"]:.0f} / 100', ra.score_label(cur["wellbeing"]) + " (composite indicator)")
    with c[2]: card("HIGH-RISK PROBABILITY", f'{p["High"]*100:.0f}%', "Model output")
    with c[3]: card("MONITORING STATUS", status, f'Trend: {trend["status"]}', ra.risk_color("High" if status == "Needs Attention" else "Moderate" if status == "Monitor Closely" else "Low"))
    ew = mon.early_warning(hist)
    if ew["active"]:
        st.markdown(f'<div class="warn">⚠ <b>Early Warning</b><br>{ew["message"]}</div>', unsafe_allow_html=True)
    st.subheader("Today's Mental Wellbeing Overview")
    a, b = st.columns(2)
    with a: show(charts.radar(ra.radar_values(cur["inputs"])))
    with b:
        st.markdown("**Indicator levels**")
        for f in ["stress", "anxiety", "mood", "social_interaction", "trauma_symptoms"]:
            st.caption(f'{LABELS[f]}: {cur["inputs"][f]:.1f} / 10'); st.progress(min(cur["inputs"][f] / 10, 1.0))
        st.caption(f'Sleep: {cur["inputs"]["sleep_hours"]:.1f} h'); st.progress(min(cur["inputs"]["sleep_hours"] / 12, 1.0))
    disclaimer()


def page_screening(bundle):
    hero("🧠 Mental Health Screening", "Provide your current wellbeing information. No names or identifying data are stored.")
    vals = {}
    with st.form("screening"):
        def sl(f):
            lab, lo, hi, d, h = FEATURES[f]
            step = 1 if f == "age" else 0.5
            return st.slider(lab, lo if f == "age" else float(lo), hi if f == "age" else float(hi),
                             d if f == "age" else float(d), step if f == "age" else float(step), help=h, key=f"in_{f}")
        st.markdown("### 👤 Personal Profile"); vals["age"] = sl("age")
        st.markdown("### 🌙 Daily Wellbeing")
        a, b = st.columns(2)
        with a: vals["sleep_hours"] = sl("sleep_hours"); vals["mood"] = sl("mood")
        with b: vals["stress"] = sl("stress"); vals["anxiety"] = sl("anxiety")
        st.markdown("### 🤝 Social & Behavioural")
        a, b, c = st.columns(3)
        with a: vals["social_interaction"] = sl("social_interaction")
        with b: vals["social_isolation"] = sl("social_isolation")
        with c: vals["activity_level"] = sl("activity_level")
        st.markdown("### 🧩 Trauma-related Indicators")
        a, b = st.columns(2)
        with a: vals["trauma_symptoms"] = sl("trauma_symptoms"); vals["intrusive_thoughts"] = sl("intrusive_thoughts")
        with b: vals["nightmares"] = sl("nightmares"); vals["hypervigilance"] = sl("hypervigilance")
        save = st.checkbox("Save this session to my local monitoring history", value=True)
        go = st.form_submit_button("ANALYZE WELLBEING", type="primary", use_container_width=True)
    if go:
        res = analyze(bundle, vals)
        st.session_state["current"] = res
        if save:
            mon.add_record(vals, res["wellbeing"], res["pred"])
        cat = res["pred"]["category"]
        st.success(f'Analysis complete: **{cat}** screening risk · wellbeing score {res["wellbeing"]:.0f}/100'
                   + (" · session saved." if save else ".") + " See **🔍 AI Risk Analysis** for the full breakdown.")
    disclaimer()


def page_trends(bundle):
    hero("📈 Monitoring & Trends", "Risk and wellbeing across your recorded sessions")
    h = mon.load_history()
    t1, t2 = st.tabs(["Trends", "Records & Data"])
    with t1:
        if h.empty:
            need_data()
        else:
            trend = mon.compute_trend(h)
            x = list(range(1, len(h) + 1))
            col = {"Improving": "#2e9e6b", "Stable": "#e0a030", "Worsening": "#d9534f"}.get(trend["status"], "#7a869a")
            a, b = st.columns([1, 2])
            with a:
                card("TREND STATUS", {"Worsening": "INCREASING RISK"}.get(trend["status"], trend["status"].upper()),
                     f'Risk slope {trend["risk_slope"]:+.1f}/session · Wellbeing {trend["wellbeing_slope"]:+.1f}/session', col)
            with b:
                st.info(trend["message"])
            ew = mon.early_warning(h)
            if ew["active"]:
                st.markdown(f'<div class="warn">⚠ <b>Early Warning</b><br>{ew["message"]}<br><small>Triggered by: '
                            f'{"; ".join(ew["triggers"])}</small></div>', unsafe_allow_html=True)
            st.markdown("**Past sessions**"); show(charts.timeline(h["predicted_risk"].tolist()))
            a, b = st.columns(2)
            with a: show(charts.trend_line(x, {"Risk score": h["risk_score"], "Wellbeing score": h["wellbeing_score"]}, "Risk & Wellbeing score", (0, 100)))
            with b: show(charts.trend_line(x, {"Stress": h["stress"], "Anxiety": h["anxiety"]}, "Stress & Anxiety", (0, 10)))
            a, b = st.columns(2)
            with a: show(charts.trend_line(x, {"Mood": h["mood"], "Social interaction": h["social_interaction"]}, "Mood & Social Interaction", (0, 10)))
            with b: show(charts.trend_line(x, {"Sleep (hours)": h["sleep_hours"]}, "Sleep"))
    with t2:
        st.dataframe(h, use_container_width=True)
        st.caption(f"Stored locally at {HISTORY_PATH}")
        with st.expander("Demo tools & maintenance"):
            if st.button("Load 4 demo sessions (simulated inputs, real model predictions)"):
                demo = [dict(stress=3, anxiety=3, mood=7, sleep_hours=7.5, trauma_symptoms=2, social_interaction=7),
                        dict(stress=4, anxiety=4, mood=6, sleep_hours=7, trauma_symptoms=3, social_interaction=6),
                        dict(stress=6, anxiety=6, mood=4.5, sleep_hours=5.5, trauma_symptoms=5, social_interaction=4),
                        dict(stress=8, anxiety=8, mood=3, sleep_hours=4.5, trauma_symptoms=7, social_interaction=2.5)]
                base = {f: FEATURES[f][3] for f in FEATURE_NAMES}
                for d in demo:
                    v = {**base, **d, "intrusive_thoughts": d["trauma_symptoms"], "nightmares": d["trauma_symptoms"],
                         "hypervigilance": d["trauma_symptoms"], "social_isolation": 10 - d["social_interaction"],
                         "activity_level": d["mood"]}
                    r = analyze(bundle, v); mon.add_record(v, r["wellbeing"], r["pred"])
                st.session_state.pop("current", None); st.rerun()
            confirm = st.checkbox("I confirm I want to permanently delete all monitoring history")
            if st.button("Clear Monitoring History", disabled=not confirm):
                mon.clear_history(); st.session_state.pop("current", None); st.rerun()
    disclaimer()


def page_analysis(bundle):
    hero("🔍 AI Risk Analysis", "Complete breakdown of the latest screening")
    cur = current(bundle)
    if cur is None:
        need_data(); disclaimer(); return
    cat, probs = cur["pred"]["category"], cur["pred"]["probabilities"]
    st.markdown(f'### Overall screening category: <span style="color:{ra.risk_color(cat)}">{cat.upper()}</span>', unsafe_allow_html=True)
    msg = {"Low": "Low screening risk. Indicators are within a typical range.",
           "Moderate": "Moderate screening risk. Further professional assessment may be appropriate if concerns persist.",
           "High": "Elevated distress risk detected. Further professional assessment may be appropriate."}[cat]
    st.write(msg)
    a, b, c = st.columns(3)
    with a: show(charts.gauge(cur["wellbeing"], "Wellbeing score (0–100)", "#2e9e6b"))
    with b: show(charts.gauge(cur["pred"]["risk_score"], "Model risk score (0–100)", ra.risk_color(cat)))
    with c: show(charts.prob_bars(probs))
    st.caption("Risk score = 50 × P(Moderate) + 100 × P(High). Wellbeing score is a separate application-level composite, not the ML output.")
    t1, t2, t3 = st.tabs(["Key Contributing Indicators", "Explainable AI", "Wellbeing score method"])
    with t1:
        st.markdown("**Indicators associated with the model's prediction** (not medically proven causes)")
        flags = ra.detect_indicators(cur["inputs"])
        if flags:
            for f, m, sev in flags:
                st.markdown(f'<span class="pill">{"●"*sev} {m} ({cur["inputs"][f]:.1f})</span>', unsafe_allow_html=True)
        else:
            st.success("No indicator is currently outside its typical range.")
    with t2:
        a, b = st.columns(2)
        with a:
            show(charts.contributions(cur["contrib"], LABELS, "Local: effect on this result"))
            st.caption("Computed by replacing each indicator with the training median and measuring the change in High-risk probability "
                       "(red = pushes risk up, green = pushes risk down). Works for any model.")
        with b:
            gi = global_importance(bundle); nm = bundle["model_name"]
            show(charts.importance(gi, LABELS, f"Global: {nm} " + ("coefficient magnitude" if nm == "Logistic Regression" else "feature importance")))
            if nm == "Logistic Regression":
                st.caption("Mean absolute coefficient across classes on standardized features; associations learned from synthetic data, not causal effects.")
    with t3:
        st.markdown("Weighted composite of normalised inputs. **Raise** score: " + ", ".join(f"{LABELS[k]} ({w:.0%})" for k, w in ra.POSITIVE.items())
                    + ". **Lower** score: " + ", ".join(f"{LABELS[k]} ({w:.0%})" for k, w in ra.NEGATIVE.items())
                    + ". Sleep is scored 100% at 7–9 h and declines outside it.")
    st.subheader("Personalized wellbeing suggestions")
    ew = mon.early_warning(mon.load_history())
    for s in suggestions(cur["inputs"], cat, ew["active"]):
        st.markdown(f"- {s}")
    disclaimer()


def page_models(bundle):
    hero("📊 Model Analytics", "Real training results from train_model.py")
    i, m, sel = bundle["info"], bundle["metrics"], bundle["model_name"]
    c = st.columns(5)
    for col, (l, v) in zip(c, [("DATASET SIZE", i["n_samples"]), ("FEATURES", i["n_features"]), ("CLASSES", i["n_classes"]),
                               ("TRAIN / TEST", f'{i["n_train"]} / {i["n_test"]}'), ("SELECTED MODEL", sel)]):
        with col: card(l, v)
    st.caption(f'Data source: {i["data_source"]} — synthetic, not real patient data. Selection used 5-fold CV macro-F1 on the training set only; '
               'metrics below are macro-averaged on the held-out test set.')
    tbl = pd.DataFrame(m).T[["accuracy", "precision", "recall", "f1", "cv_f1"]].round(3)
    tbl.columns = ["Accuracy", "Precision", "Recall", "F1", "CV F1 (train)"]
    st.dataframe(tbl, use_container_width=True)
    a, b = st.columns(2)
    with a: show(charts.model_comparison(m))
    with b: show(charts.importance(global_importance(bundle), LABELS, f"Feature importance — {sel}"))
    cols = st.columns(len(m))
    for col, (n, v) in zip(cols, m.items()):
        with col: show(charts.confusion(v["confusion_matrix"], CLASS_NAMES, n + (" (selected)" if n == sel else "")))
    disclaimer()


def page_recs(bundle):
    hero("💡 Recommendations", "Personalized wellbeing suggestions (rule-based, non-medical)")
    cur = current(bundle)
    if cur is None:
        need_data(); disclaimer(); return
    ew = mon.early_warning(mon.load_history())
    for s in suggestions(cur["inputs"], cur["pred"]["category"], ew["active"]):
        st.markdown(f"- {s}")
    st.caption("Generated by transparent Python rules in recommendations.py — not by a language model, and not medical treatment.")
    disclaimer()


def page_about(bundle):
    hero("ℹ️ About", "AI-Powered Dynamic Mental Health Monitoring and Distress Prediction System for Victims of Atrocities")
    st.markdown(f"""
**Purpose.** An academic screening and monitoring prototype that combines machine learning, wellbeing scoring, history-based trend analysis and a rule-based early-warning layer.

**Pipeline.** Synthetic dataset (4,000 records, fixed seed) → StandardScaler + classifier pipeline (Logistic Regression vs Random Forest, selected by cross-validated macro-F1) → class probabilities → risk score, wellbeing score, explanations, suggestions.

**Privacy.** Everything runs locally. Sessions are stored in `data/monitoring_history.csv` without names or identifiers.

**Limitations.** The training data is synthetic, so metrics do not reflect real-world performance. The system never diagnoses any condition.

> {DISCLAIMER}
""")


PAGES = {"🏠 Dashboard": page_dashboard, "🧠 Mental Health Screening": page_screening,
         "📈 Monitoring & Trends": page_trends, "🔍 AI Risk Analysis": page_analysis,
         "📊 Model Analytics": page_models, "💡 Recommendations": page_recs, "ℹ️ About": page_about}


def main():
    st.sidebar.title("🧠 MindMonitor AI")
    choice = st.sidebar.radio("Navigation", list(PAGES))
    st.sidebar.caption("Academic prototype · not a diagnostic tool")
    try:
        bundle = get_bundle()
    except ModelNotFoundError as e:
        st.error(str(e)); st.code("python train_model.py"); return
    PAGES[choice](bundle)


main()
