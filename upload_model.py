from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="models/sentiment_bert",
    repo_id="aliabbi/sentiment-bert",
    repo_type="model",
)
