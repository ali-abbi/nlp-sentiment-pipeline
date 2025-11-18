import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data.load_data import load_imdb_dataset


def test_load_imdb_dataset_basic():
    # Use a tiny sample for fast testing
    train_df, val_df, test_df = load_imdb_dataset(sample_size=200)

    # Basic shape checks
    assert not train_df.empty
    assert not val_df.empty
    assert not test_df.empty

    # Columns check
    for df in (train_df, val_df, test_df):
        assert "text" in df.columns
        assert "label" in df.columns

    # Label type check
    assert set(train_df["label"].unique()).issubset({0, 1})
