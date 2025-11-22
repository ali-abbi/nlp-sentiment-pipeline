from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


# Load local model
MODEL_ID = "aliabbi/sentiment-distilbert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

# Device (CPU for Docker)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)
model.eval()

app = FastAPI(title="Sentiment Analysis API")

class InputText(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
      <head>
        <title>Sentiment Analysis</title>
        <style>
          body { font-family: sans-serif; max-width: 600px; margin: 40px auto; }
          textarea { width: 100%; height: 120px; }
          button { padding: 8px 16px; margin-top: 8px; }
          .result { margin-top: 16px; font-weight: bold; }
        </style>
      </head>
      <body>
        <h1>Sentiment Analysis</h1>
        <p>Enter a review and get its sentiment.</p>
        <form method="post" action="/predict-form">
          <textarea name="text" placeholder="Type your text here..."></textarea><br/>
          <button type="submit">Analyze</button>
        </form>
      </body>
    </html>
    """

@app.post("/predict")
def predict(input: InputText):

    logger.info(f"REQUEST /predict text='{input.text[:80]}...'")

    start = time.time()

    encoded = tokenizer(
        input.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    try:
        with torch.no_grad():
            outputs = model(**encoded)
            probs = F.softmax(outputs.logits, dim=1)[0]
    except Exception as e:
        logger.error(f"🔥 MODEL ERROR: {str(e)}")
        raise e

    duration = (time.time() - start) * 1000
    logger.info(f"RESPONSE /predict took {duration:.2f}ms")

    probs = probs.cpu().numpy().tolist()
    label_id = int(torch.argmax(outputs.logits, dim=1).item())
    label = "negative" if label_id == 0 else "positive"

    return {
        "label": label,
        "probabilities": {
            "negative": probs[0],
            "positive": probs[1],
        }
    }


@app.post("/predict-form", response_class=HTMLResponse)
def predict_form(text: str = Form(...)):

    logger.info(f"REQUEST /predict-form text='{text[:80]}...'")

    start = time.time()

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    try:
        with torch.no_grad():
            outputs = model(**encoded)
            probs = F.softmax(outputs.logits, dim=1)[0]
    except Exception as e:
        logger.error(f"🔥 MODEL ERROR: {str(e)}")
        raise e

    duration = (time.time() - start) * 1000
    logger.info(f"RESPONSE /predict-form took {duration:.2f}ms")

    probs = probs.cpu().numpy().tolist()
    label_id = int(torch.argmax(outputs.logits, dim=1).item())
    label = "negative" if label_id == 0 else "positive"

    return f"""
    <html>
      <body>
        <h1>Sentiment Analysis</h1>
        <p><strong>Input:</strong> {text}</p>
        <p><strong>Prediction:</strong> {label}</p>
        <p>Negative: {probs[0]:.4f} &nbsp;&nbsp; Positive: {probs[1]:.4f}</p>
        <a href="/">Back</a>
      </body>
    </html>
    """

@app.get("/health")
def health():
    return {"status": "ok"}
