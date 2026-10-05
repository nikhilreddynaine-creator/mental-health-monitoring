"""Run: python tests/test_all.py   (or pytest tests/)"""
import sys, tempfile, importlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import config, generate_dataset, train_model, predictor, risk_analysis as ra, monitoring as mon, recommendations, charts
from config import FEATURE_NAMES, CLASS_NAMES

LOW = dict(age=30, sleep_hours=8, mood=8, stress=2, anxiety=2, social_interaction=8, social_isolation=2, activity_level=7,
           trauma_symptoms=1, intrusive_thoughts=1, nightmares=1, hypervigilance=1)
MOD = dict(age=30, sleep_hours=6, mood=5, stress=6, anxiety=5, social_interaction=5, social_isolation=5, activity_level=5,
           trauma_symptoms=5, intrusive_thoughts=5, nightmares=5, hypervigilance=5)
HIGH = dict(age=30, sleep_hours=4, mood=2, stress=9, anxiety=9, social_interaction=2, social_isolation=9, activity_level=2,
            trauma_symptoms=9, intrusive_thoughts=9, nightmares=8, hypervigilance=9)
B = predictor.load_bundle()


def test_dataset():
    a, b = generate_dataset.generate(500), generate_dataset.generate(500)
    assert a.equals(b)
    assert set(a.risk_level) == {0, 1, 2} and a.isna().sum().sum() == 0
    assert a[FEATURE_NAMES].min().min() >= 0 and a.sleep_hours.max() <= 12


def test_training_and_loading():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d); bundle = train_model.train(config.DATASET_PATH, d / "m.joblib")
        assert (d / "m.joblib").exists() and (d / "metrics.json").exists()
        for m in bundle["metrics"].values():
            assert 0.5 < m["f1"] < 1.0
        assert bundle["info"]["n_train"] + bundle["info"]["n_test"] == bundle["info"]["n_samples"]
        assert predictor.load_bundle(d / "m.joblib")["model_name"] == bundle["model_name"]


def test_missing_model():
    try:
        predictor.load_bundle(Path("/nonexistent/model.joblib")); assert False
    except predictor.ModelNotFoundError:
        pass


def test_auto_retrain_on_bad_or_missing_model():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.joblib"
        assert predictor.load_or_train(p)["model_name"]            # missing -> retrains
        p.write_bytes(b"corrupt"); assert predictor.load_or_train(p)["model_name"]   # corrupt -> retrains


def test_predictions_low_mod_high():
    p = [predictor.predict(B, x) for x in (LOW, MOD, HIGH)]
    assert [q["category"] for q in p] == ["Low", "Moderate", "High"]
    for q in p:
        assert abs(sum(q["probabilities"].values()) - 1) < 1e-6 and 0 <= q["risk_score"] <= 100
    assert p[0]["risk_score"] < p[1]["risk_score"] < p[2]["risk_score"]


def test_wellbeing_score():
    lo, mo, hi = (ra.wellbeing_score(x) for x in (LOW, MOD, HIGH))
    assert 0 <= hi < mo < lo <= 100
    assert ra.wellbeing_score({**MOD, "mood": 9}) > ra.wellbeing_score(MOD)
    assert ra.wellbeing_score({**MOD, "stress": 9}) < ra.wellbeing_score(MOD)
    assert abs(sum(ra.POSITIVE.values()) + sum(ra.NEGATIVE.values()) - 1) < 1e-9


def test_explanations_and_indicators():
    c = predictor.local_contributions(B, HIGH)
    assert set(c) == set(FEATURE_NAMES) and max(c.values()) > 0
    assert sum(predictor.global_importance(B).values()) > 0
    assert ra.detect_indicators(LOW) == [] and len(ra.detect_indicators(HIGH)) >= 6


def test_recommendations():
    assert any("professional" in s for s in recommendations.suggestions(HIGH, "High"))
    assert any("sleep schedule" in s for s in recommendations.suggestions(HIGH, "High"))
    assert any("balanced" in s for s in recommendations.suggestions(LOW, "Low"))
    txt = " ".join(recommendations.suggestions(HIGH, "High")).lower()
    assert "you have" not in txt and "mentally ill" not in txt


def test_history_and_trends():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "h.csv"
        assert mon.load_history(p).empty
        assert mon.compute_trend(mon.load_history(p))["status"] == "Insufficient data"
        p.write_text(""); assert mon.load_history(p).empty
        for x in (LOW, MOD, HIGH):
            pr = predictor.predict(B, x); df = mon.add_record(x, ra.wellbeing_score(x), pr, p)
        assert len(df) == 3 and list(mon.load_history(p).predicted_risk) == ["Low", "Moderate", "High"]
        assert mon.compute_trend(df)["status"] == "Worsening"
        assert mon.compute_trend(df.iloc[::-1].reset_index(drop=True))["status"] == "Improving"
        assert mon.compute_trend(pd.concat([df.iloc[[1]]] * 4, ignore_index=True))["status"] == "Stable"
        ew = mon.early_warning(df); assert ew["active"] and len(ew["triggers"]) >= 2
        assert not mon.early_warning(df.iloc[::-1].reset_index(drop=True))["active"]
        assert not mon.early_warning(df.iloc[:1])["active"]
        mon.clear_history(p); assert mon.load_history(p).empty


def test_charts():
    import matplotlib.pyplot as plt
    lab = {f: f for f in FEATURE_NAMES}
    figs = [charts.gauge(60, "x"), charts.radar(ra.radar_values(MOD)), charts.prob_bars({"Low": .2, "Moderate": .5, "High": .3}),
            charts.contributions(predictor.local_contributions(B, MOD), lab, "t"),
            charts.importance(predictor.global_importance(B), lab, "t"),
            charts.trend_line([1, 2, 3], {"a": [1, 2, 3]}, "t"), charts.timeline(["Low", "High"]),
            charts.confusion(B["metrics"][B["model_name"]]["confusion_matrix"], CLASS_NAMES, "t"),
            charts.model_comparison(B["metrics"])]
    assert len(figs) == 9; plt.close("all")


class Ctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, n): return lambda *a, **k: Ctx()
    def __call__(self, *a, **k): return Ctx()


class FakeSt:
    """Minimal stand-in for streamlit so app.py's page code executes without the package."""
    def __init__(self, submit=False, click=False):
        self.session_state = {}; self.submit, self.click = submit, click; self.calls = []; self.sidebar = self
    def __getattr__(self, n):
        self.calls.append(n); return lambda *a, **k: Ctx()
    def cache_resource(self, f): return f
    def set_page_config(self, **k): pass
    def columns(self, n, **k): return [Ctx() for _ in range(n if isinstance(n, int) else len(n))]
    def tabs(self, l): return [Ctx() for _ in l]
    def slider(self, label, lo, hi, d, *a, **k): return d
    def checkbox(self, *a, **k): return True
    def form_submit_button(self, *a, **k): return self.submit
    def button(self, label="", *a, **k): return self.click and label.startswith("Load")
    def radio(self, label, opts, **k): return opts[0]
    def form(self, *a, **k): return Ctx()
    def expander(self, *a, **k): return Ctx()
    def pyplot(self, fig, **k): self.calls.append("pyplot")
    def rerun(self): pass


class SessionDict(dict):
    pass


def run_app(fake):
    class Mod(type(sys)): pass
    m = Mod("streamlit")
    m.__getattr__ = None
    sys.modules["streamlit"] = fake
    sys.modules.pop("app", None)
    return importlib.import_module("app")


def test_streamlit_pages_with_stub():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "h.csv"; orig = config.HISTORY_PATH
        saved = (mon.load_history.__defaults__, mon.add_record.__defaults__, mon.clear_history.__defaults__)
        mon.load_history.__defaults__ = (tmp,); mon.add_record.__defaults__ = (tmp, None); mon.clear_history.__defaults__ = (tmp,)
        try:
            app = run_app(FakeSt())
            for fn in app.PAGES.values(): fn(B)                      # empty history, every page
            f2 = FakeSt(submit=True); app = run_app(f2)
            app.page_screening(B); assert len(mon.load_history()) == 1 and f2.session_state["current"]
            f3 = FakeSt(click=True); app = run_app(f3)
            app.page_trends(B); assert len(mon.load_history()) >= 5  # demo sessions added
            for fn in app.PAGES.values(): fn(B)                      # populated history, every page
            assert f3.calls.count("pyplot") > 15
        finally:
            mon.load_history.__defaults__, mon.add_record.__defaults__, mon.clear_history.__defaults__ = saved
            sys.modules.pop("streamlit", None); sys.modules.pop("app", None)


if __name__ == "__main__":
    fails = 0
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            try: f(); print("PASS", n)
            except Exception:
                fails += 1; import traceback; print("FAIL", n); traceback.print_exc()
    print("ALL PASSED" if not fails else f"{fails} FAILED"); sys.exit(1 if fails else 0)
