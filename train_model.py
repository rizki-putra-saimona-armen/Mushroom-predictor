"""
train_model.py
----------------
Melatih ulang model klasifikasi jamur (edible / poisonous) persis mengikuti
pipeline pada notebook `Part_7_-_Latihan.ipynb`:

  1. Buang kolom `veil_type` (nilainya konstan, tidak informatif)
  2. Pilih 4 fitur dengan korelasi terkuat terhadap target (dari association
     matrix): odor, gill_color, ring_type, spore_print_color
  3. One-hot encode ke-4 fitur kategorik tersebut
  4. OneVsRestClassifier(LogisticRegression) + GridSearchCV
     (param_grid identik dengan notebook)

Dipecah jadi dua fungsi:
  - build_and_fit()  -> murni di memori, TIDAK menyentuh disk sama sekali.
                        Aman dipanggil di filesystem read-only (mis. Vercel).
  - train_and_save() -> build_and_fit() lalu tulis ke model/*.{pkl,json}.
                        Dipakai untuk CLI / cache lokal.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "mushrooms.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "mushroom_model.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"

FEATURES = ["odor", "gill_color", "ring_type", "spore_print_color"]
TARGET = "edible"


def build_and_fit():
    """Melatih pipeline murni di memori (tidak menulis apa pun ke disk).
    Return (fitted_estimator, metadata_dict)."""
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["veil_type"])

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer([
        ("categoric", OneHotEncoder(handle_unknown="ignore"), FEATURES),
    ])

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("algo", OneVsRestClassifier(
            LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
        )),
    ])

    param_grid = {
        "algo__estimator__C": [0.1, 1, 10],
        "algo__estimator__fit_intercept": [True, False],
    }

    # n_jobs=1 (bukan -1): GridSearchCV n_jobs=-1 memakai multiprocessing
    # (loky), yang di banyak sandbox serverless (Vercel dkk.) tidak diizinkan
    # spawn proses baru dan bisa ikut menyebabkan crash. Dataset ini kecil
    # (8rb baris, 18 fit), jadi n_jobs=1 tetap cepat (~2-5 detik).
    grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=1, verbose=1)
    grid.fit(X_train, y_train)

    train_acc = grid.score(X_train, y_train)
    cv_best = grid.best_score_
    test_acc = grid.score(X_test, y_test)

    y_pred = grid.predict(X_test)
    y_proba = grid.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)

    print("Best params :", grid.best_params_)
    print(f"Train acc   : {train_acc:.4f}")
    print(f"CV best     : {cv_best:.4f}")
    print(f"Test acc    : {test_acc:.4f}")
    print(f"ROC AUC     : {auc:.4f}")

    dropdown_options = {
        feat: sorted(df[feat].dropna().unique().tolist()) for feat in FEATURES
    }

    metadata = {
        "features": FEATURES,
        "options": dropdown_options,
        "metrics": {
            "train_accuracy": round(train_acc, 4),
            "cv_best_score": round(cv_best, 4),
            "test_accuracy": round(test_acc, 4),
            "roc_auc": round(auc, 4),
            "precision_edible": round(report["True"]["precision"], 4),
            "recall_edible": round(report["True"]["recall"], 4),
            "n_samples": int(len(df)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        },
        "best_params": {k: str(v) for k, v in grid.best_params_.items()},
        "sklearn_version": sklearn.__version__,
    }

    return grid.best_estimator_, metadata


def train_and_save():
    """build_and_fit() lalu simpan ke model/mushroom_model.pkl +
    model/metadata.json. Dipakai untuk cache lokal / CLI. JANGAN dipakai
    di filesystem read-only — pakai build_and_fit() saja di situ."""
    estimator, metadata = build_and_fit()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, MODEL_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nModel disimpan -> {MODEL_PATH}")
    print(f"Metadata disimpan -> {METADATA_PATH}")
    return estimator, metadata


if __name__ == "__main__":
    train_and_save()
