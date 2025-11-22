# 🧠 Sentiment Analysis Pipeline

**Custom-trained DistilBERT model • FastAPI • Docker • Render Deployment**

A production-ready NLP pipeline for sentiment classification (positive/negative) trained on IMDB reviews.  
Built end-to-end with:

- **Dataset loading & preprocessing**
- **Custom fine-tuned DistilBERT**
- **FastAPI inference service**
- **Dockerized deployment**
- **HuggingFace model hosting**
- **Live Render API service**

---

## 🚀 Live Demo

### 🌐 Web App (HTML Form)
https://nlp-sentiment-pipeline.onrender.com  

### 📘 API Docs
https://nlp-sentiment-pipeline.onrender.com/docs

---

## 🏗 Architecture Overview

```
                ┌────────────────────┐
                │  Training Pipeline │
                │ ─────────────────── │
                │  • Load IMDB data  │
                │  • Clean text      │
                │  • Tokenize        │
                │  • Fine-tune       │
                │    DistilBERT      │
                │  • Save model      │
                └─────────┬──────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │   HuggingFace Hub      │
              │  Stores model + tokenizer
              └─────────┬──────────────┘
                          │ pulls
                          ▼
       ┌────────────────────────────────────┐
       │              FastAPI               │
       │  • /predict        (JSON API)      │
       │  • /predict-form  (HTML UI)        │
       │  • /health         (health check)  │
       └───────────┬────────────────────────┘
                   │ Docker container
                   ▼
    ┌──────────────────────────────────────────┐
    │                Render.com                │
    │   • CPU-only container runtime           │
    │   • Auto-redeploy on git push            │
    │   • Public URL hosting                   │
    └──────────────────────────────────────────┘
```

---

## 📦 Project Structure

```
nlp-sentiment-pipeline/
│
├── api/
│   └── app.py              # FastAPI inference server
│
├── src/
│   ├── data/load_data.py   # Dataset loading
│   ├── utils/text_cleaning.py
│   └── models/train_distilbert.py
│
├── models/                 # (empty locally — model on HuggingFace)
│
├── tests/                  # pytest suite
│
├── Dockerfile              # Deployment container
├── requirements.txt
├── start.sh
└── README.md               # this file
```

---

## 🔥 Training Details

### **Model**
`distilbert-base-uncased` → fine-tuned for binary sentiment classification  
Uploaded to HuggingFace: https://huggingface.co/aliabbi/sentiment-distilbert

### **Dataset**
IMDB movie reviews (positive/negative)

### **Metrics**
| Metric | Score |
|--------|--------|
| Accuracy | **0.88** |
| F1-score | **0.88** |

---

## 🧪 Run Locally

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run FastAPI
```bash
uvicorn api.app:app --reload
```

### Run with Docker
```bash
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```

---

## 📡 API Usage

### POST `/predict`

#### Request:
```json
{
  "text": "I loved this movie!"
}
```

#### Response:
```json
{
  "label": "positive",
  "probabilities": {
    "negative": 0.02,
    "positive": 0.98
  }
}
```

---

## 🎨 Web UI

The `/predict-form` route renders a simple UI that includes:

- Text input form  
- Sentiment label  
- Confidence percentages  
- Colored confidence bars  

---

## 🧪 Testing

Run the full test suite:

```bash
pytest -q
```

Covers:

- Data loading  
- Text cleaning  
- API responses  

---

## 📄 License

MIT License © 2025 aliabbi
