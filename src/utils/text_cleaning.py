import re

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = text.replace("\n", " ")
    text = text.strip()
    return text
