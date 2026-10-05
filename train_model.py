"""Train Logistic Regression & Random Forest, select by CV macro-F1 (train only), evaluate on held-out test."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import CLASS_NAMES, DATASET_PATH, FEATURE_NAMES, MODEL_PATH, SEED
from generate_dataset import generate


def build(clf):
    pre = ColumnTransformer([("num", StandardScaler(), FEATURE_NAMES)])
    return Pipeline([("pre", pre), ("clf", clf)])


def train(dataset_path=DATASET_PATH, model_path=MODEL_PATH, test_size=0.2):
    if not dataset_path.exists():
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        generate().to_csv(dataset_path, index=False)
    df = pd.read_csv(dataset_path)
    X, y = df[FEATURE_NAMES], df["risk_level"]
    # split BEFORE any fitting (scaler lives inside the pipeline -> no leakage)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=SEED)

    candidates = {
        "Logistic Regression": build(LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)),
        "Random Forest": build(RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5,
                                                      random_state=SEED, n_jobs=-1)),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    results, fitted = {}, {}
    for name, pipe in candidates.items():
        cv_f1 = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="f1_macro").mean()
        pipe.fit(Xtr, ytr)
        p = pipe.predict(Xte)
        results[name] = dict(
            accuracy=accuracy_score(yte, p),
            precision=precision_score(yte, p, average="macro"),
            recall=recall_score(yte, p, average="macro"),
            f1=f1_score(yte, p, average="macro"),
            cv_f1=cv_f1,
            confusion_matrix=confusion_matrix(yte, p).tolist(),
        )
        fitted[name] = pipe
    best = max(results, key=lambda k: results[k]["cv_f1"])  # selection uses CV on train only

    importances = {}
    rf = fitted["Random Forest"].named_steps["clf"]
    importances["Random Forest"] = dict(zip(FEATURE_NAMES, rf.feature_importances_.tolist()))
    lr = fitted["Logistic Regression"].named_steps["clf"]
    importances["Logistic Regression"] = dict(zip(FEATURE_NAMES, np.abs(lr.coef_).mean(axis=0).tolist()))
    lr_high = dict(zip(FEATURE_NAMES, lr.coef_[2].tolist()))

    bundle = dict(
        pipeline=fitted[best], model_name=best, feature_names=FEATURE_NAMES, class_names=CLASS_NAMES,
        medians=Xtr.median().to_dict(), metrics=results, importances=importances, lr_high_coefs=lr_high,
        info=dict(n_samples=len(df), n_features=len(FEATURE_NAMES), n_classes=3,
                  n_train=len(Xtr), n_test=len(Xte), class_counts=y.value_counts().sort_index().tolist(),
                  data_source="Synthetic (seed=%d)" % SEED),
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    (model_path.parent / "metrics.json").write_text(json.dumps(
        dict(selected=best, metrics=results, info=bundle["info"]), indent=2), encoding="utf-8")
    return bundle


if __name__ == "__main__":
    b = train()
    print("Selected:", b["model_name"])
    for k, v in b["metrics"].items():
        print(f"{k:20s} acc={v['accuracy']:.3f} prec={v['precision']:.3f} rec={v['recall']:.3f} "
              f"f1={v['f1']:.3f} cvF1={v['cv_f1']:.3f}")
