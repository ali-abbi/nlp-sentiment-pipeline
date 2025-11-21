from huggingface_hub import HfApi, upload_folder
import os

MODEL_PATH = "models/sentiment_distilbert"
HF_REPO = "aliabbi/sentiment-distilbert"

HF_TOKEN = os.getenv("HF_TOKEN")

api = HfApi()

upload_folder(
    folder_path=MODEL_PATH,
    repo_id=HF_REPO,
    repo_type="model",
    token=HF_TOKEN
)

print("Model uploaded successfully!")
