# src/merge_and_retrain.py
import os
import sqlite3
import subprocess
import pandas as pd

TRAIN_PATH = "data/processed/train.csv"
NEW_DATA_PATH = "data/processed/new_production_labeled.csv"
DB_PATH = "data/production_logs.db"

def clear_processed_production_logs():
    """Flushes the local SQLite tracking tables to clear the volume gate metric windows."""
    if not os.path.exists(DB_PATH):
        return
        
    print("\n🧹 Cleaning up production logs database table for the next batch cycle...")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM inference_logs")
        conn.commit()
        conn.close()
        print("✨ Production database cleared and reset to 0 rows. Volume gate initialized.")
    except Exception as e:
        print(f"⚠️ Non-breaking database cleanup warnings: {str(e)}")

def merge_and_version_dataset():
    print("🔄 Initializing Dataset Merge Engine (v1 ➡️ v2)...")
    
    if not os.path.exists(NEW_DATA_PATH):
        print(f"❌ Aborting: No newly labeled data found at {NEW_DATA_PATH}. Run orchestrator.py first.")
        return

    # 1. Load historical train matrix split
    old_train_df = pd.read_csv(TRAIN_PATH)
    print(f"   Baseline training dataset records: {len(old_train_df)}")

    # 2. Ingest structured LLM-labeled data
    new_labeled_df = pd.read_csv(NEW_DATA_PATH)
    print(f"   Newly logged production records: {len(new_labeled_df)}")

    # 3. Concatenate safely over unified schemas
    updated_train_df = pd.concat([old_train_df, new_labeled_df], ignore_index=True)
    print(f"   🚀 Blended Dataset v2 total count: {len(updated_train_df)}")

    # 4. Save file payload over old pointers
    updated_train_df.to_csv(TRAIN_PATH, index=False)
    print(f"💾 Dataset v2 successfully written to: {TRAIN_PATH}")

    # 5. Automatically version file state with Data Version Control (DVC)
    print("🔒 Versioning updated files with Data Version Control (DVC)...")
    try:
        subprocess.run(["dvc", "add", "data/processed"], check=True)
        print("✅ DVC updated tracker pointer schemas successfully.")
        
        # 6. Flush active table states to close loop processes
        clear_processed_production_logs()
        
        # Clean up temporary batch artifacts
        if os.path.exists(NEW_DATA_PATH):
            os.remove(NEW_DATA_PATH)
            
        print("\n💡 NEXT METRIC STEP: Run 'git add data/processed.dvc' and commit your update files.")
        print("🏋️ Push to GitHub using commit tag 'trigger-retrain' to scale up continuous model retraining jobs!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Data lineage version control execution failed: {e}")

if __name__ == "__main__":
    merge_and_version_dataset()