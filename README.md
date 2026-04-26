# 🎬 CineGenre — Movie Genre Predictor
## NLP Project | Flask Backend + HTML Frontend

---

## 📁 Project Structure

```
your-project-folder/
│
├── app.py                  ← Flask backend (your NLP model)
├── requirements.txt        ← Python dependencies
├── static/
│   └── index.html          ← Frontend (served by Flask)
└── dataset/
    ├── movies_meta.csv      ← Download from Kaggle (see below)
    └── movies_subtitles.csv ← Download from Kaggle (see below)
```

---

## ⚙️ Setup Instructions

### Step 1 — Download the Dataset
1. Go to: https://www.kaggle.com/datasets/adiamaan/movie-subtitle-dataset
2. Download and extract it
3. Copy `movies_meta.csv` and `movies_subtitles.csv` into a folder called `dataset/`
   (next to app.py)

> If your CSV files are somewhere else, open `app.py` and change:
> ```python
> DATASET_PATH = "./dataset"   # ← change this path
> ```

---

### Step 2 — Install Python dependencies
```bash
pip install -r requirements.txt
```

---

### Step 3 — Run the backend
```bash
python app.py
```

The server starts on http://localhost:5000
The model **trains automatically in the background** — this takes 3–10 minutes
depending on your CPU. The frontend shows a live progress status.

---

### Step 4 — Open the frontend
Open your browser and go to:
```
http://localhost:5000
```

The page will show a loading status bar.
Once the model is ready, the **Predict Genre** button activates automatically.

---

## 🧠 How It Works

| Step | What happens |
|------|-------------|
| 1 | Load `movies_meta.csv` + `movies_subtitles.csv` |
| 2 | Merge on `imdb_id` |
| 3 | Clean text (lowercase, remove punctuation) |
| 4 | Tokenise with NLTK `word_tokenize` |
| 5 | Remove English stopwords |
| 6 | Lemmatise (verb + noun) with WordNetLemmatizer |
| 7 | Group by film, build `text_for_tfidf` |
| 8 | Fit TF-IDF Vectoriser (10,000 features, 1–2 ngrams) |
| 9 | Binarise multi-labels with `MultiLabelBinarizer` |
| 10 | Train `OneVsRestClassifier(LogisticRegression)` |
| 11 | Evaluate: accuracy, hamming loss, per-genre F1 |

The frontend calls `/api/predict` with the subtitle text and threshold.
The backend runs the same preprocessing pipeline and returns genre predictions.

---

## 🔗 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves `index.html` |
| `/api/status` | GET | Returns model training status |
| `/api/predict` | POST | Predicts genres from text |
| `/api/metrics` | GET | Returns model performance metrics |

### POST `/api/predict` — Request body
```json
{
  "text": "subtitle text here...",
  "threshold": 0.2,
  "is_srt": false
}
```

### POST `/api/predict` — Response
```json
{
  "genres": ["Action", "Thriller"],
  "confidence": { "Action": 0.85, "Thriller": 0.62, ... },
  "top5": [["Action", 0.85], ["Thriller", 0.62], ...],
  "word_count": 120,
  "threshold": 0.2
}
```

---

## 🛠 Troubleshooting

**"Cannot reach backend"** → Make sure `python app.py` is running.

**"Dataset CSVs not found"** → Check that `DATASET_PATH` in `app.py` points to
the folder containing both CSV files.

**Model training seems stuck** → Check your terminal — it prints live progress.
Lemmatisation on large datasets is slow; wait 5–10 minutes.

**Port 5000 in use** → Change `app.run(port=5000)` at the bottom of `app.py`
to another port like `5001`, then visit `http://localhost:5001`.
