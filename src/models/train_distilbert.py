import os
from typing import Dict, Any
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
from datasets import Dataset

from src.data.load_data import load_imdb_dataset


def preprocess_data(tokenizer, df):
    encodings = tokenizer(
        df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=256,
    )
    encodings["labels"] = df["label"].tolist()
    return encodings


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


def train_model(
    model_name="distilbert-base-uncased",
    output_dir="models/sentiment_distilbert",
    sample_size=None,
    epochs=1,
    batch_size=8,
    debug=True,
):
    # Detect device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n🔥 Using device: {device.upper()}\n")

    # ---------------------------------------------------------
    # STEP 1 — Load data
    # ---------------------------------------------------------
    train_df, val_df, test_df = load_imdb_dataset(sample_size=sample_size)

    # ---------------------------------------------------------
    # STEP 2 — Tokenization debug (optional)
    # ---------------------------------------------------------
    if debug:
        print("\n=== STEP 2: TOKENIZATION DEBUG ===\n")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encodings = preprocess_data(tokenizer, train_df.head(10))
        print("Encoding keys:", encodings.keys())
        print("Num samples:", len(encodings["input_ids"]))
        print("input_ids[0] length:", len(encodings["input_ids"][0]))
        print("Example label:", encodings["labels"][0])
        print("Tokenization OK.\n")

    # ---------------------------------------------------------
    # STEP 3 — Create datasets
    # ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_enc = preprocess_data(tokenizer, train_df)
    val_enc = preprocess_data(tokenizer, val_df)
    test_enc = preprocess_data(tokenizer, test_df)

    train_dataset = Dataset.from_dict(train_enc)
    val_dataset = Dataset.from_dict(val_enc)
    test_dataset = Dataset.from_dict(test_enc)

    # ---------------------------------------------------------
    # STEP 3 — DataLoader / batch debug
    # ---------------------------------------------------------
    if debug:
        print("\n=== STEP 3: DATALOADER DEBUG ===\n")

        from torch.utils.data import DataLoader

        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=default_data_collator,
        )

        batch = next(iter(train_loader))
        print("Batch keys:", batch.keys())
        print("input_ids shape:", batch["input_ids"].shape)
        print("attention_mask shape:", batch["attention_mask"].shape)
        print("labels shape:", batch["labels"].shape)

    # ---------------------------------------------------------
    # STEP 4 — Forward pass debug
    # ---------------------------------------------------------
    if debug:
        print("\n=== STEP 4: FORWARD PASS DEBUG ===\n")

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        ).to(device)

        # Move batch to correct device
        for k in batch:
            batch[k] = batch[k].to(device)

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        print("Loss:", outputs.loss.item())
        print("Logits shape:", outputs.logits.shape)
        print("Logits sample:", outputs.logits[0].detach().cpu().numpy())
        print("\nForward pass OK.\n")

        print("🚀 DEBUGGING COMPLETE. TRAINING IS CURRENTLY DISABLED.\n")
        return  # STOP HERE during debugging

    # ---------------------------------------------------------
    # STEP 5 — REAL TRAINING (enabled only when debug=False)
    # ---------------------------------------------------------

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        max_eval_samples=200 if sample_size is not None else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
    )

    trainer.train()

    # Evaluate
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    print("\nTest Set Performance:")
    print(classification_report(test_df["label"], preds))

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nModel saved to {output_dir}")
    return trainer, model, tokenizer


# -------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------
if __name__ == "__main__":
    # Debug mode enabled → NO TRAINING WILL RUN
    train_model(sample_size=20, epochs=1, batch_size=4, debug=True)

    # When ready for real training:
    # train_model(sample_size=None, epochs=2, batch_size=16, debug=False)
