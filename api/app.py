from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load local model
MODEL_PATH = "models/sentiment_distilbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Device (CPU for Docker)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)
model.eval()

app = FastAPI(title="Sentiment Analysis API")

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(input: InputText):
    encoded = tokenizer(
        input.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        probs = F.softmax(outputs.logits, dim=1)[0].cpu().tolist()

    # Use logits argmax, not list argmax
    label_id = int(torch.argmax(outputs.logits, dim=1).item())
    label = "negative" if label_id == 0 else "positive"

    return {
        "label": label,
        "probabilities": {
            "negative": probs[0],
            "positive": probs[1],
        }
    }
