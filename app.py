"""
app.py — MycoLab: Klasifikasi Jamur (Edible vs Poisonous)

Menyajikan antarmuka web untuk model dari Part_7_-_Latihan.ipynb
(OneVsRestClassifier(LogisticRegression) di atas 4 fitur kategorik).

PENTING soal model: file .pkl TIDAK dikirim bersama proyek ini. Model
dilatih otomatis saat startup, memakai scikit-learn yang terpasang di
tempat app ini dijalankan — supaya tidak ada masalah kompatibilitas
pickle lintas versi/instalasi.

Penyimpanan ke disk (model/*.pkl) sifatnya best-effort, BUKAN wajib:
platform serverless seperti Vercel punya filesystem read-only (kecuali
/tmp), jadi kalau proses tulis gagal, app tetap jalan normal memakai
model yang sudah ada di memori — hanya saja start berikutnya (cold
start baru) akan melatih ulang lagi. Ini supaya app tidak pernah crash
gara-gara tidak bisa menulis file.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "mushroom_model.pkl"
METADATA_PATH = BASE_DIR / "model" / "metadata.json"

print(f"[MycoLab] app.py — scikit-learn {sklearn.__version__}")

# static_folder="public" + static_url_path="": di Vercel, file statis HARUS
# ada di folder public/** untuk dilayani lewat CDN (bukan lewat Flask
# static_folder bawaan). Dengan konfigurasi ini, url_for('static', ...)
# tetap menghasilkan path yang benar di kedua environment (lokal & Vercel).
app = Flask(__name__, static_folder="public", static_url_path="")


def _load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _smoke_test(m, meta):
    """Coba satu prediksi nyata. Melempar exception kalau pickle/model
    tidak kompatibel dengan scikit-learn yang sedang berjalan."""
    probe = pd.DataFrame([{k: v[0] for k, v in meta["options"].items()}])
    m.predict_proba(probe)


def _try_cache_to_disk(m, meta):
    """Simpan model & metadata ke disk kalau memungkinkan. Best-effort —
    kalau filesystem read-only (Vercel dkk.), cukup catat lalu lanjut,
    JANGAN sampai membuat app crash karena ini."""
    try:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(m, MODEL_PATH)
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print("[MycoLab] Model di-cache ke disk untuk start berikutnya.")
    except OSError as exc:
        print(f"[MycoLab] Filesystem read-only (lumrah di serverless "
              f"seperti Vercel) — jalan memakai model in-memory saja, "
              f"tanpa cache disk: {exc}")


def load_or_train_model():
    # 1) Coba pakai model yang sudah ter-cache di disk (kalau ada & valid).
    #    Ini juga aman di filesystem read-only karena cuma proses BACA.
    if MODEL_PATH.exists() and METADATA_PATH.exists():
        try:
            m = joblib.load(MODEL_PATH)
            meta = _load_metadata()
            _smoke_test(m, meta)
            print("[MycoLab] Model tersimpan valid, langsung dipakai.")
            return m, meta
        except Exception as exc:  # noqa: BLE001
            print(f"[MycoLab] Model tersimpan tidak kompatibel "
                  f"({exc.__class__.__name__}: {exc}).")

    # 2) Tidak ada model valid -> latih di memori (tidak menyentuh disk).
    print("[MycoLab] Melatih model baru di memori memakai scikit-learn "
          f"lokal ({sklearn.__version__})... (~5-10 detik)")
    from train_model import build_and_fit
    m, meta = build_and_fit()
    _smoke_test(m, meta)
    print("[MycoLab] Model baru berhasil dilatih & terverifikasi.")

    # 3) Coba simpan ke disk sebagai cache untuk start berikutnya.
    #    Kalau gagal (read-only fs), tidak apa — model di memori tetap dipakai.
    _try_cache_to_disk(m, meta)

    return m, meta


# ---- load model & metadata sekali saat startup -----------------------
model, METADATA = load_or_train_model()

FEATURES = METADATA["features"]
OPTIONS = METADATA["options"]
METRICS = METADATA["metrics"]

# Label tampilan yang lebih ramah manusia untuk tiap nilai kategori
LABELS = {
    "odor": {
        "almond": "Almond", "anise": "Adas (anise)", "creosote": "Creosote",
        "fishy": "Amis", "foul": "Busuk", "musty": "Apek",
        "none": "Tidak berbau", "pungent": "Menyengat", "spicy": "Pedas",
    },
    "gill_color": {
        "black": "Hitam", "brown": "Cokelat", "buff": "Krem",
        "chocolate": "Cokelat tua", "gray": "Abu-abu", "green": "Hijau",
        "orange": "Oranye", "pink": "Merah muda", "purple": "Ungu",
        "red": "Merah", "white": "Putih", "yellow": "Kuning",
    },
    "ring_type": {
        "evanescent": "Evanescent (mudah hilang)", "flaring": "Melebar",
        "large": "Besar", "none": "Tidak ada", "pendant": "Menggantung",
    },
    "spore_print_color": {
        "black": "Hitam", "brown": "Cokelat", "buff": "Krem",
        "chocolate": "Cokelat tua", "green": "Hijau", "orange": "Oranye",
        "purple": "Ungu", "white": "Putih", "yellow": "Kuning",
    },
}

FEATURE_META = [
    {"key": "odor", "label": "Bau (Odor)", "hint": "Aroma tudung/batang jamur"},
    {"key": "gill_color", "label": "Warna Insang", "hint": "Warna bilah di bawah tudung"},
    {"key": "ring_type", "label": "Tipe Cincin", "hint": "Bentuk cincin pada batang"},
    {"key": "spore_print_color", "label": "Warna Spora", "hint": "Warna jejak spora di kertas"},
]


@app.route("/")
def index():
    fields = []
    for meta in FEATURE_META:
        key = meta["key"]
        opts = [{"value": v, "label": LABELS[key].get(v, v)} for v in OPTIONS[key]]
        fields.append({**meta, "options": opts})

    return render_template("index.html", fields=fields, metrics=METRICS)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}

    missing = [f for f in FEATURES if not payload.get(f)]
    if missing:
        return jsonify({
            "error": f"Data belum lengkap: {', '.join(missing)}"
        }), 400

    row = {f: payload[f] for f in FEATURES}
    X = pd.DataFrame([row], columns=FEATURES)

    try:
        proba = model.predict_proba(X)[0]
        pred = model.predict(X)[0]
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Gagal melakukan prediksi: {exc}"}), 400

    classes = list(model.classes_)  # [False, True]
    proba_map = {str(c): float(p) for c, p in zip(classes, proba)}
    proba_edible = proba_map.get("True", 0.0)
    proba_poisonous = proba_map.get("False", 0.0)

    is_edible = bool(pred)
    confidence = proba_edible if is_edible else proba_poisonous

    return jsonify({
        "prediction": "edible" if is_edible else "poisonous",
        "confidence": round(confidence * 100, 2),
        "proba_edible": round(proba_edible * 100, 2),
        "proba_poisonous": round(proba_poisonous * 100, 2),
        "input": {f: LABELS[f].get(row[f], row[f]) for f in FEATURES},
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
