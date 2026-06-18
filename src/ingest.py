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
    
    print("🔄 Aggregating annotator responses per unique comment using IRT scores...")
    # Step 1: Average the IRT-adjusted hate_speech_score for each unique comment
    grouped = df.groupby('comment_id').agg({
        'text': 'first',
        'hate_speech_score': 'mean'
    }).reset_index()
    
    # Step 2: Convert continuous average IRT rating into strict binary label 
    # (> 0.5 is hate speech based on dataset recommendations)
    grouped['is_hate_speech'] = (grouped['hate_speech_score'] > 0.5).astype(int)
    
    # Drop the continuous score column to keep the schema clean for training
    grouped = grouped.drop(columns=['hate_speech_score'])
        
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