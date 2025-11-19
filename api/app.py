from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load model + tokenizer
MODEL_DIR = "models/sentiment_bert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# Detect device (MPS for M1/M2 Macs)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)

app = FastAPI(title="Sentiment Analysis API")

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(input: InputText):
    # Tokenize input
    encoded = tokenizer(
        input.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    # Move tensors to device
    encoded = {k: v.to(device) for k, v in encoded.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)[0]

    # Convert to Python type
    probs = probs.cpu().numpy().tolist()
    label_id = int(torch.argmax(logits, dim=1).cpu().item())
    label = "negative" if label_id == 0 else "positive"

    return {
        "label": label,
        "probabilities": {
            "negative": probs[0],
            "positive": probs[1],
        }
    }
