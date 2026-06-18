# src/ingest.py
import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def load_and_aggregate_data():
    print("📥 Loading dataset from Hugging Face...")
    dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech", split="train")
    df = dataset.to_pandas()
    
    print(f"Raw rows loaded: {len(df)}")
    print("🔄 Aggregating data into Multi-Label Schema...")

    # Step 1: Define the core dimensions we want DistilBERT to predict
    target_labels = ['hatespeech', 'insult', 'humiliate', 'dehumanize', 'violence']

    # Step 2: Average both the IRT score and the raw sub-labels per unique comment
    agg_dict = {'text': 'first', 'hate_speech_score': 'mean'}
    for label in target_labels:
        agg_dict[label] = 'mean'

    grouped = df.groupby('comment_id').agg(agg_dict).reset_index()

    # Step 3: Binarize the labels
    # - For IRT hate_speech_score, > 0.5 means Unambiguous Hate Speech
    grouped['label_hate_speech'] = (grouped['hate_speech_score'] > 0.5).astype(int)
    
    # - For ordinal sub-labels (0-4 scale), an average > 1.5 indicates strong agreement among annotators
    grouped['label_insult'] = (grouped['insult'] > 1.5).astype(int)
    grouped['label_humiliate'] = (grouped['humiliate'] > 1.5).astype(int)
    grouped['label_dehumanize'] = (grouped['dehumanize'] > 1.5).astype(int)
    grouped['label_violence'] = (grouped['violence'] > 1.5).astype(int)

    # Step 4: Drop the intermediary raw columns to keep the dataset clean
    columns_to_drop = ['hate_speech_score'] + target_labels
    grouped = grouped.drop(columns=columns_to_drop)

    print(f"Unique consolidated comments: {len(grouped)}")
    return grouped

def split_and_save(df):
    print("✂️ Splitting data into Train (80%), Val (10%), Test (10%)...")
    
    # Clean text columns slightly (remove extreme trailing white spaces)
    df['text'] = df['text'].str.strip()
    df = df.dropna(subset=['text'])

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)
    
    # Create directory if it doesn't exist
    os.makedirs("data/processed", exist_ok=True)
    
    # Save snapshots
    train_df.to_csv("data/processed/train.csv", index=False)
    val_df.to_csv("data/processed/val.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    
    print("✅ Data successfully processed and saved to data/processed/")

if __name__ == "__main__":
    cleaned_df = load_and_aggregate_data()
    split_and_save(cleaned_df)