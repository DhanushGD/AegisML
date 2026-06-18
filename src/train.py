# src/train.py
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score, roc_auc_score

# 1. Custom Multi-Label Dataset Class
class ToxicDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Multi-label classification requires FloatTensor for target values
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

# 2. Multi-Label Evaluation Metrics Strategy
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Apply sigmoid over raw logits to get confidence scores between 0 and 1
    sigmoid = 1 / (1 + np.exp(-predictions))
    # Threshold at 0.5 to get binary predictions per label
    preds = (sigmoid >= 0.5).astype(int)
    
    # Calculate Macro F1 (perfect for class imbalance)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    
    # Calculate Macro ROC-AUC
    try:
        macro_auc = roc_auc_score(labels, sigmoid, average='macro')
    except ValueError:
        macro_auc = 0.0 # Handle edge cases where a label might have only 1 class in a small batch
        
    return {
        'macro_f1': macro_f1,
        'macro_auc': macro_auc
    }

def train():
    print("🚀 Starting Training Infrastructure Script...")
    
    # Define paths
    train_path = "data/processed/train.csv"
    val_path = "data/processed/val.csv"
    
    if not os.path.exists(train_path):
        raise FileNotFoundError("❌ Preprocessed data not found! Did you run 'dvc pull'?")
        
    # Load dataset splits
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    target_cols = ['hatespeech', 'offensive', 'dehumanizing', 'violence', 'genocide']
    num_labels = len(target_cols)
    
    # Initialize DistilBERT Tokenizer
    print("📥 Loading DistilBERT Tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    # Tokenize input texts
    print("🔤 Tokenizing dataset splits...")
    train_encodings = tokenizer(list(train_df['text'].astype(str)), truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(list(val_df['text'].astype(str)), truncation=True, padding=True, max_length=128)
    
    # Extract label arrays
    train_labels = train_df[target_cols].values
    val_labels = val_df[target_cols].values
    
    # Create PyTorch datasets
    train_dataset = ToxicDataset(train_encodings, train_labels)
    val_dataset = ToxicDataset(val_encodings, val_labels)
    
    # Initialize Model with a Multi-Label Sequence Classification Head
    print("🤖 Initializing DistilBertForSequenceClassification head...")
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', 
        num_labels=num_labels,
        problem_type="multi_label_classification" # Tells Hugging Face to use BCEWithLogitsLoss
    )
    
    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=torch.cuda.is_available(), # Use mixed-precision training if a GPU is active
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("🏋️ Training starting now...")
    trainer.train()
    
    # Save the model artifacts locally
    print("💾 Saving fine-tuned model artifacts...")
    model.save_pretrained("./models/toxic_distilbert")
    tokenizer.save_pretrained("./models/toxic_distilbert")
    print("✅ Training complete! Model saved to ./models/toxic_distilbert")

if __name__ == "__main__":
    train()