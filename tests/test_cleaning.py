import sys
from pathlib import Path

# Add project root to PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.utils.text_cleaning import clean_text


def test_clean_text_basic():
    assert clean_text("Hello WORLD") == "hello world"


def test_clean_text_newline():
    assert clean_text("Hello\nWorld") == "hello world"

