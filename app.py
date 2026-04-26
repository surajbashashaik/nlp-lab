# ============================================================
#  CineGenre - Movie Genre Predictor
#  Flask Backend  |  app.py
#  Run: python app.py
#  Then open: http://localhost:5000
# ============================================================

import os
import re
import ast
import threading
import logging

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, hamming_loss, accuracy_score

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────
DATASET_PATH  = "./dataset"   # folder with movies_meta.csv + movies_subtitles.csv
STATIC_FOLDER = "./static"    # folder with index.html

app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path="")
CORS(app)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  Global model state
# ─────────────────────────────────────────────
model_state = {
    "status":  "loading",
    "message": "Initialising…",
    "model":   None,
    "tfidf":   None,
    "mlb":     None,
    "metrics": {}
}

# ─────────────────────────────────────────────
#  NLTK downloads
# ─────────────────────────────────────────────
def download_nltk():
    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
        nltk.download(pkg, quiet=True)

# ─────────────────────────────────────────────
#  Genre parsing  ← THE KEY FIX
#  The dataset stores genres as JSON strings like:
#  "[{'id': 28, 'name': 'Action'}, {'id': 35, 'name': 'Comedy'}]"
#  We parse them properly to get clean names.
# ─────────────────────────────────────────────
def parse_genres(genres_val):
    if pd.isna(genres_val) or str(genres_val).strip() in ("", "[]"):
        return []
    genres_str = str(genres_val).strip()

    # Try ast.literal_eval for list-of-dicts format
    if genres_str.startswith("["):
        try:
            parsed = ast.literal_eval(genres_str)
            if isinstance(parsed, list):
                names = []
                for item in parsed:
                    if isinstance(item, dict) and "name" in item:
                        names.append(item["name"].strip())
                    elif isinstance(item, str):
                        names.append(item.strip())
                return [n for n in names if n]
        except Exception:
            pass
        # Regex fallback
        names = re.findall(r"'name'\s*:\s*'([^']+)'", genres_str)
        if names:
            return names

    # Plain comma-separated fallback
    return [g.strip() for g in genres_str.split(",") if g.strip()]

# ─────────────────────────────────────────────
#  Text preprocessing
# ─────────────────────────────────────────────
_stop_words = None
_lemmatizer = None

def get_nlp_tools():
    global _stop_words, _lemmatizer
    if _stop_words is None:
        _stop_words = set(stopwords.words("english"))
        _lemmatizer = WordNetLemmatizer()
    return _stop_words, _lemmatizer


def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r"[^a-z\u00c0-\u017f ]", "", text)
        return text
    return ""


def preprocess(text):
    stop_words, lemmatizer = get_nlp_tools()
    text = text.lower()
    text = re.sub(r"[^a-z ]", "", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    lemmas = []
    for w in tokens:
        lemma = lemmatizer.lemmatize(w, pos="v")
        if lemma == w:
            lemma = lemmatizer.lemmatize(w, pos="n")
        lemmas.append(lemma)
    return " ".join(lemmas)


def clean_srt(text):
    text = re.sub(r"\n\d+\n", "\n", text)
    text = re.sub(r"\d{2}:\d{2}:\d{2},\d{3} --> .*", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n+", " ", text)
    return text.strip()


# ─────────────────────────────────────────────
#  Model training
# ─────────────────────────────────────────────
def train_model():
    global model_state
    try:
        download_nltk()

        # ── Load CSVs ──────────────────────────────────────────────
        model_state["message"] = "Loading dataset…"
        meta_path      = os.path.join(DATASET_PATH, "movies_meta.csv")
        subtitles_path = os.path.join(DATASET_PATH, "movies_subtitles.csv")

        if not os.path.exists(meta_path) or not os.path.exists(subtitles_path):
            raise FileNotFoundError(
                f"Dataset CSVs not found in '{DATASET_PATH}'. "
                "Download from https://www.kaggle.com/datasets/adiamaan/movie-subtitle-dataset"
            )

        meta_df      = pd.read_csv(meta_path, low_memory=False)
        subtitles_df = pd.read_csv(subtitles_path, low_memory=False)
        log.info(f"Loaded meta: {meta_df.shape}  subtitles: {subtitles_df.shape}")

        # ── Parse genres BEFORE merge ──────────────────────────────
        model_state["message"] = "Parsing genres…"
        log.info(f"Sample genres value: {repr(meta_df['genres'].iloc[0])}")

        meta_df["genre_list"] = meta_df["genres"].apply(parse_genres)
        meta_df = meta_df[meta_df["genre_list"].map(len) > 0].copy()
        log.info(f"Meta after genre filter: {meta_df.shape}")

        all_genres = sorted(set(g for gl in meta_df["genre_list"] for g in gl))
        log.info(f"Unique genres ({len(all_genres)}): {all_genres}")

        # ── Merge ──────────────────────────────────────────────────
        model_state["message"] = "Merging dataframes…"
        merged_df = pd.merge(meta_df, subtitles_df, on="imdb_id", how="inner")
        log.info(f"Merged shape: {merged_df.shape}")

        # ── Clean text ─────────────────────────────────────────────
        model_state["message"] = "Cleaning text…"
        merged_df["cleaned_text"] = merged_df["text"].apply(clean_text)

        # ── Tokenise + stopwords + lemmatise ───────────────────────
        model_state["message"] = "Tokenising & lemmatising (several minutes)…"
        stop_words, lemmatizer = get_nlp_tools()

        def lemmatize_tokens(tokens):
            lemmas = []
            for w in tokens:
                lemma = lemmatizer.lemmatize(w, pos="v")
                if lemma == w:
                    lemma = lemmatizer.lemmatize(w, pos="n")
                lemmas.append(lemma)
            return lemmas

        merged_df["tokens"]            = merged_df["cleaned_text"].apply(word_tokenize)
        merged_df["filtered_tokens"]   = merged_df["tokens"].apply(
            lambda toks: [w for w in toks if w not in stop_words])
        merged_df["lemmatized_tokens"] = merged_df["filtered_tokens"].apply(lemmatize_tokens)

        # ── Group by film ──────────────────────────────────────────
        model_state["message"] = "Grouping by film…"
        text_grouped = (
            merged_df
            .groupby("imdb_id")["lemmatized_tokens"]
            .apply(lambda x: [w for toks in x for w in toks])
            .reset_index()
        )
        text_grouped.rename(columns={"lemmatized_tokens": "all_tokens"}, inplace=True)

        film_genres = meta_df[["imdb_id", "genre_list"]].drop_duplicates("imdb_id")
        film_df = pd.merge(text_grouped, film_genres, on="imdb_id", how="inner")
        film_df["text_for_tfidf"] = film_df["all_tokens"].apply(lambda x: " ".join(x))
        film_df = film_df[film_df["text_for_tfidf"].str.strip() != ""].copy()
        log.info(f"Film-level rows: {film_df.shape[0]}")

        # ── TF-IDF ─────────────────────────────────────────────────
        model_state["message"] = "Fitting TF-IDF vectoriser…"
        tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=5, max_df=0.8)
        X = tfidf.fit_transform(film_df["text_for_tfidf"])
        log.info(f"TF-IDF shape: {X.shape}")

        # ── Multi-label binariser ──────────────────────────────────
        mlb = MultiLabelBinarizer()
        y   = mlb.fit_transform(film_df["genre_list"])
        log.info(f"Genre classes ({len(mlb.classes_)}): {list(mlb.classes_)}")

        # ── Train / test split ─────────────────────────────────────
        model_state["message"] = "Training classifier…"
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        clf = OneVsRestClassifier(LogisticRegression(max_iter=2000))
        clf.fit(X_train, y_train)
        log.info("Model trained ✅")

        # ── Evaluate ───────────────────────────────────────────────
        model_state["message"] = "Evaluating model…"
        y_pred = clf.predict(X_test)
        report = classification_report(
            y_test, y_pred,
            target_names=mlb.classes_,
            output_dict=True,
            zero_division=0
        )
        acc = round(accuracy_score(y_test, y_pred), 4)
        hl  = round(hamming_loss(y_test, y_pred), 4)

        metrics = {
            "accuracy":     acc,
            "hamming_loss": hl,
            "genres":       list(mlb.classes_),
            "per_genre": {
                g: {
                    "precision": round(report[g]["precision"], 3),
                    "recall":    round(report[g]["recall"],    3),
                    "f1":        round(report[g]["f1-score"],  3),
                }
                for g in mlb.classes_ if g in report
            }
        }

        model_state.update({
            "status":  "ready",
            "message": "Model ready · TF-IDF + Logistic Regression",
            "model":   clf,
            "tfidf":   tfidf,
            "mlb":     mlb,
            "metrics": metrics,
        })
        log.info(f"🎬 CineGenre ready! Accuracy={acc}  HammingLoss={hl}")

    except Exception as e:
        log.error(f"Training failed: {e}", exc_info=True)
        model_state.update({"status": "error", "message": str(e)})


# ─────────────────────────────────────────────
#  API Routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(STATIC_FOLDER, "index.html")


@app.route("/api/status")
def status():
    return jsonify({"status": model_state["status"], "message": model_state["message"]})


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        if model_state["status"] != "ready":
            return jsonify({"error": "Model not ready yet. Please wait."}), 503

        data      = request.get_json(force=True)
        text      = data.get("text", "").strip()
        threshold = float(data.get("threshold", 0.2))
        is_srt    = bool(data.get("is_srt", False))

        if not text:
            return jsonify({"error": "No text provided."}), 400

        if is_srt:
            text = clean_srt(text)

        processed = preprocess(text)
        if not processed.strip():
            return jsonify({"error": "Text too short — no meaningful words after preprocessing."}), 400

        clf   = model_state["model"]
        tfidf = model_state["tfidf"]
        mlb   = model_state["mlb"]

        # ───── Prediction ─────
        vector = tfidf.transform([processed])
        probs  = clf.predict_proba(vector)

        # Handle different sklearn outputs safely
        if isinstance(probs, list):
            prob_values = np.array([
                p[1] if isinstance(p, (list, np.ndarray)) and len(p) > 1 else p[0]
                for p in probs
            ])
        else:
            prob_values = probs[0]   # correct shape (n_classes,)

        prob_values = np.array(prob_values).flatten()

        # ───── Thresholding ─────
        pred_mask = (prob_values >= threshold).astype(int)

        # Safety check
        if pred_mask.shape[0] != len(mlb.classes_):
            raise ValueError(
                f"Shape mismatch: got {pred_mask.shape[0]} probs but expected {len(mlb.classes_)}"
            )

        # ───── Convert to labels ─────
        genres_predicted = list(
            mlb.inverse_transform(pred_mask.reshape(1, -1))[0]
        )

        # ───── Confidence scores ─────
        confidence = {
            genre: round(float(prob_values[i]), 3)
            for i, genre in enumerate(mlb.classes_)
        }

        top5 = sorted(confidence.items(), key=lambda x: x[1], reverse=True)[:5]

        return jsonify({
            "genres":     genres_predicted,
            "confidence": confidence,
            "top5":       top5,
            "word_count": len(processed.split()),
            "threshold":  threshold,
        })

    except Exception as e:
        log.error(f"Predict error: {e}", exc_info=True)
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500


@app.route("/api/metrics")
def metrics():
    if model_state["status"] != "ready":
        return jsonify({"error": "Model not ready."}), 503
    return jsonify(model_state["metrics"])


# ─────────────────────────────────────────────
#  Startup
# ─────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Starting model training in background…")
    t = threading.Thread(target=train_model, daemon=True)
    t.start()
    app.run(debug=False, port=5000, use_reloader=False)