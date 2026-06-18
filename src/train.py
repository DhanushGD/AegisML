# src/train.py

# ============================================================
# MULTI-LABEL TOXIC CONTENT CLASSIFICATION USING DISTILBERT
# ============================================================
#
# Goal:
# Given a piece of text, predict multiple toxicity labels:
#
#   hatespeech
#   offensive
#   dehumanizing
#   violence
#   genocide
#
# IMPORTANT:
# This is NOT Multi-Class Classification.
#
# Multi-Class:
#   Choose ONE class from many.
#
#   Example:
#   [0, 0, 1, 0, 0]
#
# Multi-Label:
#   Multiple labels can be true simultaneously.
#
#   Example:
#   [1, 1, 0, 1, 0]
#
# Therefore:
# - We use BCEWithLogitsLoss
# - We use Sigmoid activation
# - We predict one probability per label
#
# HuggingFace automatically handles BCEWithLogitsLoss
# because we specify:
#
# problem_type="multi_label_classification"
#
# ============================================================

import os
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from sklearn.metrics import (
    f1_score,
    roc_auc_score
)

# ============================================================
# DATASET CLASS
# ============================================================
#
# HuggingFace Trainer expects data in this format:
#
# {
#     "input_ids": tensor(...),
#     "attention_mask": tensor(...),
#     "labels": tensor([...])
# }
#
# The tokenizer generates:
#
# {
#     "input_ids": [...],
#     "attention_mask": [...]
# }
#
# This Dataset class simply packages:
#
#   tokenized inputs
#   +
#   corresponding labels
#
# into the format expected by Trainer.
#
# Example output:
#
# {
#     "input_ids": tensor(...),
#     "attention_mask": tensor(...),
#     "labels": tensor([1,0,1,0,0])
# }
#
# Why float labels?
#
# BCEWithLogitsLoss requires float targets:
#
# Correct:
#   [1.0, 0.0, 1.0, 0.0, 0.0]
#
# Not:
#   [1, 0, 1, 0, 0]
#
# ============================================================

class ToxicDataset(Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):

        # Fetch tokenized inputs
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }

        # Multi-label targets must be float tensors
        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.float
        )

        return item

    def __len__(self):
        return len(self.labels)


# ============================================================
# EVALUATION METRICS
# ============================================================
#
# During training the model outputs RAW LOGITS.
#
# Example:
#
# logits =
# [2.3, 1.7, -0.4, -2.1, -3.2]
#
# These are NOT probabilities.
#
# We apply SIGMOID:
#
# probability = 1 / (1 + e^(-logit))
#
# Result:
#
# [0.91, 0.85, 0.40, 0.11, 0.04]
#
# Then threshold at 0.5:
#
# >= 0.5 → 1
# < 0.5  → 0
#
# Final prediction:
#
# [1,1,0,0,0]
#
# We compute:
#
# 1. Macro F1
#    - treats every label equally
#    - good for class imbalance
#
# 2. Macro ROC-AUC
#    - measures ranking quality
#    - uses probabilities rather than binary labels
#
# ============================================================

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    # Convert logits → probabilities
    sigmoid = 1 / (1 + np.exp(-predictions))

    # Convert probabilities → binary predictions
    preds = (sigmoid >= 0.5).astype(int)

    # Macro F1 score
    macro_f1 = f1_score(
        labels,
        preds,
        average="macro",
        zero_division=0
    )

    # Macro ROC-AUC
    try:
        macro_auc = roc_auc_score(
            labels,
            sigmoid,
            average="macro"
        )
    except ValueError:
        macro_auc = 0.0

    return {
        "macro_f1": macro_f1,
        "macro_auc": macro_auc
    }


# ============================================================
# TRAINING PIPELINE
# ============================================================
#
# Training flow:
#
# Text
#  ↓
# Tokenizer
#  ↓
# input_ids
#  ↓
# DistilBERT Encoder
#  ↓
# 768-dimensional semantic representation
#  ↓
# Classification Head (Linear Layer)
#  ↓
# 5 logits
#  ↓
# BCEWithLogitsLoss
#  ↓
# Backpropagation
#  ↓
# Weight Updates
#
# ============================================================

def train():

    print("🚀 Starting Training Infrastructure Script...")

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    train_path = "data/processed/train.csv"
    val_path = "data/processed/val.csv"

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            "❌ Preprocessed data not found! "
            "Did you run 'dvc pull'?"
        )

    # --------------------------------------------------------
    # Load Dataset
    # --------------------------------------------------------

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # --------------------------------------------------------
    # Target Labels
    # --------------------------------------------------------

    target_cols = [
        "hatespeech",
        "offensive",
        "dehumanizing",
        "violence",
        "genocide"
    ]

    num_labels = len(target_cols)

    print(f"📌 Number of labels: {num_labels}")

    # --------------------------------------------------------
    # Load Tokenizer
    # --------------------------------------------------------
    #
    # DistilBERT tokenizer converts text into:
    #
    # "I hate them"
    #
    # ↓
    #
    # ["i", "hate", "them"]
    #
    # ↓
    #
    # [1045, 5223, 2068]
    #
    # --------------------------------------------------------

    print("📥 Loading DistilBERT Tokenizer...")

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        "distilbert-base-uncased"
    )

    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------
    #
    # truncation=True
    #     Cut text longer than max_length
    #
    # padding=True
    #     Pad shorter sequences
    #
    # max_length=128
    #     Maximum sequence length
    #
    # --------------------------------------------------------

    print("🔤 Tokenizing dataset splits...")

    train_encodings = tokenizer(
        list(train_df["text"].astype(str)),
        truncation=True,
        padding=True,
        max_length=128
    )

    val_encodings = tokenizer(
        list(val_df["text"].astype(str)),
        truncation=True,
        padding=True,
        max_length=128
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    train_labels = train_df[target_cols].values
    val_labels = val_df[target_cols].values

    # --------------------------------------------------------
    # Build Dataset Objects
    # --------------------------------------------------------

    train_dataset = ToxicDataset(
        train_encodings,
        train_labels
    )

    val_dataset = ToxicDataset(
        val_encodings,
        val_labels
    )

    # --------------------------------------------------------
    # Model Initialization
    # --------------------------------------------------------
    #
    # DistilBERT Architecture:
    #
    # Input Text
    #      ↓
    # DistilBERT Encoder
    #      ↓
    # 768-D Hidden Representation
    #      ↓
    # Linear Layer
    #      ↓
    # 5 Logits
    #
    # problem_type="multi_label_classification"
    #
    # tells HuggingFace to use:
    #
    # BCEWithLogitsLoss
    #
    # instead of:
    #
    # CrossEntropyLoss
    #
    # --------------------------------------------------------

    print("🤖 Initializing model...")

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )

    # --------------------------------------------------------
    # Training Arguments
    # --------------------------------------------------------

    training_args = TrainingArguments(

        output_dir="./results",

        num_train_epochs=3,

        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,

        warmup_steps=100,

        weight_decay=0.01,

        logging_dir="./logs",
        logging_steps=50,

        evaluation_strategy="epoch",

        save_strategy="epoch",

        load_best_model_at_end=True,

        metric_for_best_model="macro_f1",

        # Mixed precision training on GPU
        fp16=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------
    #
    # Trainer automatically performs:
    #
    # for each batch:
    #
    #   forward pass
    #   compute loss
    #   backward pass
    #   optimizer step
    #   scheduler step
    #
    # --------------------------------------------------------

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # --------------------------------------------------------
    # Training Starts
    # --------------------------------------------------------

    print("🏋️ Training starting now...")

    trainer.train()

    # --------------------------------------------------------
    # Save Model
    # --------------------------------------------------------
    #
    # Saves:
    #
    # model weights
    # tokenizer vocab
    # config
    #
    # for future inference
    #
    # --------------------------------------------------------

    print("💾 Saving fine-tuned model artifacts...")

    model.save_pretrained(
        "./models/toxic_distilbert"
    )

    tokenizer.save_pretrained(
        "./models/toxic_distilbert"
    )

    print(
        "✅ Training complete! "
        "Model saved to ./models/toxic_distilbert"
    )


if __name__ == "__main__":
    train()

# ============================================================
# HOW DOES THE MODEL LEARN HATEFULNESS?
# ============================================================
#
# DistilBERT is already pre-trained on massive text corpora.
#
# During fine-tuning:
#
# Text:
#   "They should all be eliminated"
#
# Label:
#   [1,1,1,1,0]
#
# The model predicts probabilities.
#
# If predictions are wrong:
#
# BCEWithLogitsLoss computes error.
#
# Backpropagation updates weights so that
# similar future sentences produce better predictions.
#
# Over thousands of examples the model learns patterns such as:
#
# "kill them"
# "subhuman"
# "animals"
# "wipe them out"
#
# and associates them with specific toxicity labels.
#
# The model is NOT using keyword matching.
#
# It learns contextual semantic patterns from examples.
# ============================================================