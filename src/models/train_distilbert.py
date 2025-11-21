import os
import numpy as np
import torch

from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from src.data.load_data import load_imdb_dataset


# --------------------------
# DEVICE SELECTION
# --------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🔥 Using device: MPS")
else:
    device = torch.device("cpu")
    print("🔥 Using device: CPU")


# --------------------------
# DATA PREPROCESSING
# --------------------------
def preprocess_data(tokenizer, df):
    encodings = tokenizer(
        df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=256,
    )
    encodings["labels"] = df["label"].tolist()
    return encodings


# --------------------------
# METRICS
# --------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


# --------------------------
# MAIN TRAINING FUNCTION
# --------------------------
def train_model(
    model_name="distilbert-base-uncased",
    output_dir="models/sentiment_distilbert",
    sample_size=None,
    epochs=2,
    batch_size=16,
    debug=False,
):
    # 1. Load dataset
    train_df, val_df, test_df = load_imdb_dataset(sample_size=sample_size)

    # 2. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3. Tokenize
    train_enc = preprocess_data(tokenizer, train_df)
    val_enc = preprocess_data(tokenizer, val_df)
    test_enc = preprocess_data(tokenizer, test_df)

    # Convert to HF Dataset
    train_dataset = Dataset.from_dict(train_enc)
    val_dataset = Dataset.from_dict(val_enc)
    test_dataset = Dataset.from_dict(test_enc)

    # 4. Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device)

    # 5. Training args
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
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

    # 7. Train
    trainer.train()

    # 8. Evaluate
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)

    print("\n=== Test Set Performance ===")
    print(classification_report(test_df["label"], preds))

    # 9. Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n✅ Model saved to: {output_dir}")

    return trainer, model, tokenizer


# =====================================================================
# MAIN ENTRY WITH ARGPARSE — DEBUG & TRAINING MODES
# =====================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DistilBERT sentiment model.")

    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training.")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Use a subset of the dataset (int or None).")

    parser.add_argument("--debug", action="store_true",
                        help="Run debug mode (NO training).")

    args = parser.parse_args()

    # ------------------------------------------------
    # 🔍 DEBUG MODE — NO TRAINING WILL RUN
    # ------------------------------------------------
    if args.debug:
        print("\n⚠️ DEBUG MODE ENABLED — TRAINING IS DISABLED.\n")

        # 1) Tokenization Debug
        print("\n=== STEP 2: TOKENIZATION DEBUG ===")
        train_df, _, _ = load_imdb_dataset(sample_size=20)
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        encodings = preprocess_data(tokenizer, train_df)

        print("Encoding keys:", encodings.keys())
        print("Num samples:", len(encodings["input_ids"]))
        print("input_ids[0] length:", len(encodings["input_ids"][0]))
        print("Example label:", encodings["labels"][0])
        print("Tokenization OK.\n")

        # 2) Dataloader Debug
        print("\n=== STEP 3: DATALOADER DEBUG ===")
        train_dataset = Dataset.from_dict(encodings)
        from torch.utils.data import DataLoader
        loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                            collate_fn=default_data_collator)
        batch = next(iter(loader))
        print("Batch keys:", batch.keys())
        print("input_ids shape:", batch["input_ids"].shape)
        print("attention_mask shape:", batch["attention_mask"].shape)
        print("labels shape:", batch["labels"].shape)

        # 3) Forward Pass Debug
        print("\n=== STEP 4: FORWARD PASS DEBUG ===")
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        ).to(device)

        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )

        print("Loss:", float(outputs.loss))
        print("Logits shape:", outputs.logits.shape)
        print("Logits sample:", outputs.logits[0].detach().cpu().numpy())
        print("\n🚀 DEBUGGING COMPLETE. TRAINING IS DISABLED.\n")
        exit()

    # ------------------------------------------------
    # 🟢 REAL TRAINING MODE
    # ------------------------------------------------
    print("\n🚀 Starting REAL training...\n")

    train_model(
        model_name="distilbert-base-uncased",
        sample_size=args.sample_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        debug=False,
    )
