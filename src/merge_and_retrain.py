# src/merge_and_retrain.py
import pandas as pd
import subprocess
import os

TRAIN_PATH = "data/processed/train.csv"
NEW_DATA_PATH = "data/processed/new_production_labeled.csv"

def merge_and_version_dataset():
    print("🔄 Initializing Dataset Merge Engine (v1 ➡️ v2)...")
    
    if not os.path.exists(NEW_DATA_PATH):
        print(f"❌ Aborting: No newly labeled data found at {NEW_DATA_PATH}.")
        return

    # 1. Load your original training matrix
    old_train_df = pd.read_csv(TRAIN_PATH)
    print(f"   Baseline training dataset records: {len(old_train_df)}")

    # 2. Load the production data annotated by Llama 3
    new_labeled_df = pd.read_csv(NEW_DATA_PATH)
    print(f"   Newly logged production records: {len(new_labeled_df)}")

    # 3. Concatenate them together (schemas match perfectly due to our structural contract!)
    updated_train_df = pd.concat([old_train_df, new_labeled_df], ignore_index=True)
    print(f"   🚀 Blended Dataset v2 total count: {len(updated_train_df)}")

    # 4. Overwrite train.csv with our updated v2 data matrix
    updated_train_df.to_csv(TRAIN_PATH, index=False)
    print(f"💾 Dataset v2 successfully written to {TRAIN_PATH}")

    # 5. Automatically execute DVC commands to capture the new data snapshot
    print("🔒 Versioning updated files with Data Version Control (DVC)...")
    try:
        # Run 'dvc add data/processed' programmatically 
        subprocess.run(["dvc", "add", "data/processed"], check=True)
        print("✅ DVC updated pointer file successfully.")
        
        print("\n💡 NEXT STEP: Run 'git add data/processed.dvc' and commit code changes.")
        print("🏋️ Then, push your code to GitHub and trigger your Google Colab retraining script to fine-tune on Dataset V2!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ DVC command execution failed: {e}")

if __name__ == "__main__":
    merge_and_version_dataset()