"""
Task 5: Pre-trained Language Model

This script fine-tunes DistilBERT for sentiment classification.

It does two experiments:
1. Multiclass classification: negative / neutral / positive
2. Binary classification: negative / positive
"""

from pathlib import Path
import os
import random


# ---------------------------------------------------------------------------
# Basic settings
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
DATA_PATH = RESULTS_DIR / "preprocessed_dataset.csv"
CACHE_DIR = ROOT / "hf_cache"

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 64
BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 2e-5
RANDOM_SEED = 42

RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Store downloaded transformer files inside the project folder.
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(CACHE_DIR / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR / "transformers")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"


import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging


# Make the output easier to read.
logging.set_verbosity_error()

# Make results more reproducible.
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

# Turns texts and labels into the format DistilBERT needs.
class TextDataset(Dataset):

    def __init__(self, texts, labels, tokenizer):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": torch.tensor(label),
        }

# Train the model once over the training data.
def train_model(model, train_loader, optimizer, device):
    
    model.train()

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = output.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Use the trained model to predict labels for the test data.
def test_model(model, test_loader, device):
    
    model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = output.logits.argmax(dim=1).cpu().tolist()

            predicted_labels.extend(predictions)
            true_labels.extend(batch["labels"].tolist())

    return true_labels, predicted_labels

# Run one complete transformer experiment.
def run_experiment(data, title):
    
    print_section(title)

    # Convert text labels such as "positive" into numbers such as 0, 1, 2.
    label_names = sorted(data["label"].unique())
    label_to_number = {label: number for number, label in enumerate(label_names)}
    number_to_label = {number: label for label, number in label_to_number.items()}

    texts = data["sentence"]
    labels = data["label"].map(label_to_number)

    # Use the same 80/20 split idea as Task 3.
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_data = TextDataset(X_train, y_train, tokenizer)
    test_data = TextDataset(X_test, y_test, tokenizer)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print("Model:", MODEL_NAME)
    print("Device:", device)
    print("Labels:", label_names)
    print("Training rows:", len(X_train))
    print("Test rows:", len(X_test))

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_names),
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        print("Training epoch:", epoch + 1)
        train_model(model, train_loader, optimizer, device)

    true_numbers, predicted_numbers = test_model(model, test_loader, device)

    true_text_labels = [number_to_label[number] for number in true_numbers]
    predicted_text_labels = [number_to_label[number] for number in predicted_numbers]

    accuracy = accuracy_score(true_text_labels, predicted_text_labels)
    macro_f1 = f1_score(true_text_labels, predicted_text_labels, average="macro")

    print("Accuracy:", accuracy)
    print("Macro F1:", macro_f1)
    print()
    print(classification_report(true_text_labels, predicted_text_labels))

    return {
        "experiment": title,
        "model": MODEL_NAME,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "labels": ", ".join(label_names),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_length": MAX_LENGTH,
    }


# ---------------------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------------------

print_section("1. Load the data")

all_data = pd.read_csv(DATA_PATH)
all_data = all_data[["sentence", "label"]].dropna()

print("Rows:", len(all_data))
print(all_data["label"].value_counts())


# ---------------------------------------------------------------------------
# 2. Run multiclass classification
# ---------------------------------------------------------------------------

multiclass_result = run_experiment(
    all_data,
    "2. Transformer multiclass: negative / neutral / positive",
)


# ---------------------------------------------------------------------------
# 3. Run binary classification
# ---------------------------------------------------------------------------

binary_data = all_data[all_data["label"] != "neutral"].copy()

binary_result = run_experiment(
    binary_data,
    "3. Transformer binary: negative / positive",
)


# ---------------------------------------------------------------------------
# 4. Save the results
# ---------------------------------------------------------------------------

summary = pd.DataFrame([multiclass_result, binary_result])
summary.to_csv(RESULTS_DIR / "task5_transformer_summary.csv", index=False)

print_section("4. Save the results")
print("Saved:", RESULTS_DIR / "task5_transformer_summary.csv")
