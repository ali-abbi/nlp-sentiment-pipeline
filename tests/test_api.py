import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_predict_endpoint():
    response = client.post(
        "/predict",
        json={"text": "This movie was amazing!"}
    )

    assert response.status_code == 200
    data = response.json()

    assert "label" in data
    assert "probabilities" in data
    assert "positive" in data["probabilities"]
    assert "negative" in data["probabilities"]
