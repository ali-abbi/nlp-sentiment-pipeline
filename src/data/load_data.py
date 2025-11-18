from typing import Tuple, Optional

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.utils.text_cleaning import clean_text


def load_imdb_dataset(
    sample_size: Optional[int] = None,
    val_size: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the IMDb sentiment dataset, clean the text, and split into
    train / validation / test DataFrames.

    Args:
        sample_size: if not None, use only the first `sample_size` rows of the training set
                     (useful for quick experiments).
        val_size: fraction of the training data to use for validation.
        seed: random seed for reproducible splits.

    Returns:
        train_df, val_df, test_df: pandas DataFrames with columns ["text", "label"].
    """
    # Load dataset from Hugging Face (will download on first run)
    dataset = load_dataset("imdb")

    # Convert to pandas
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()

    # Optional downsampling for quick experiments
    if sample_size is not None:
        train_df = train_df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    # Keep only the columns we care about and rename them for consistency
    train_df = train_df[["text", "label"]].copy()
    test_df = test_df[["text", "label"]].copy()

    # Clean text
    train_df["text"] = train_df["text"].apply(clean_text)
    test_df["text"] = test_df["text"].apply(clean_text)

    # Split train into train/validation
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        random_state=seed,
        stratify=train_df["label"],
    )

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df
