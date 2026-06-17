# src/ingest.py
import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def load_and_aggregate_data():
    print("📥 Loading dataset from Hugging Face...")
    # Load the dataset into a pandas dataframe
    dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech", split="train")
    df = dataset.to_pandas()
    
    print(f"Raw rows loaded: {len(df)}")
    
    # Target columns we want to predict as multi-label binary targets
    target_cols = ['hatespeech', 'insult', 'dehumanize', 'violence', 'genocide']
    
    # Step 1: Average the annotator scores for each comment
    # The raw values are typically continuous scores or probabilities from annotators
    print("🔄 Aggregating annotator responses per unique comment...")
    grouped = df.groupby('comment_id').agg({
        'text': 'first',
        'hatespeech': 'mean',
        'insult': 'mean',
        'dehumanize': 'mean',
        'violence': 'mean',
        'genocide': 'mean'
    }).reset_index()
    
    # Step 2: Convert continuous average ratings into strict binary 0 or 1 labels 
    # (e.g., if mean score > 0.5, we classify it as true for that label)
    for col in target_cols:
        grouped[col] = (grouped[col] >= 0.5).astype(int)
        
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