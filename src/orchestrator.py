# src/orchestrator.py
import os
import sqlite3
import json
import pandas as pd
import requests
import ollama
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping

# Configuration Parameters
LIVE_SPACE_DB_URL = "https://your-hf-space-url/download_logs"  # Replace with your actual HF Space URL
DB_PATH = "data/production_logs.db"
VAL_PATH = "data/processed/val.csv"
DRIFT_THRESHOLD = 0.50          # 50% threshold metric
BATCH_VOLUME_THRESHOLD = 1000    # 1000-row volume gate metric

def download_latest_production_logs():
    print(f"📡 Syncing logs from production cloud registry: {LIVE_SPACE_DB_URL}...")
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        response = requests.get(LIVE_SPACE_DB_URL, timeout=30)
        if response.status_code == 200:
            with open(DB_PATH, "wb") as f:
                f.write(response.content)
            print("📥 Live production database successfully synchronized locally.")
            return True
        else:
            print(f"⚠️ Failed to pull production database. Server responded with: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Network infrastructure connection error: {str(e)}")
        return False

def check_drift_and_slice():
    print("🚀 Stage 7 Orchestrator: Initializing operational evaluation...")
    
    # Always attempt a fresh database download from production first
    download_success = download_latest_production_logs()
    
    if not os.path.exists(DB_PATH):
        print("❌ Operational Halt: No production database log file found.")
        return

    # ==========================================================
    # 🛑 THE VOLUME GATEKEEPER CHECK
    # ==========================================================
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Safely verify the table exists before querying row volume counts
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='inference_logs'")
    if not cursor.fetchone():
        print("ℹ️ Table 'inference_logs' does not exist yet. No entries recorded.")
        conn.close()
        return

    cursor.execute("SELECT COUNT(*) FROM inference_logs")
    total_logs = cursor.fetchone()[0]
    conn.close()
    
    print(f"📦 Production Data Status: {total_logs}/{BATCH_VOLUME_THRESHOLD} rows accumulated.")
    
    if total_logs < BATCH_VOLUME_THRESHOLD:
        print(f"😴 Batch size insufficient. Postponing analytics loops until {BATCH_VOLUME_THRESHOLD} logs scale up.")
        return
        
    print("🔥 Threshold breached! Evaluating text distribution matrices...")

    # 1. Gather Reference vs Current Data
    ref_df = pd.read_csv(VAL_PATH)
    
    conn = sqlite3.connect(DB_PATH)
    current_df = pd.read_sql_query("""
        SELECT input_text as text, label_hate_speech, label_insult, label_humiliate, label_dehumanize, label_violence 
        FROM inference_logs
    """, conn)
    conn.close()
    
    # 2. Build Column Matrix Feature Metadata
    column_mapping = ColumnMapping()
    column_mapping.text_features = ['text']
    column_mapping.numerical_features = ['label_hate_speech', 'label_insult', 'label_humiliate', 'label_dehumanize', 'label_violence']
    
    # 3. Compute Data Drift using Dictionary evaluation
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=ref_df, current_data=current_df, column_mapping=column_mapping)
    
    # Extract structural metrics programmatically
    result_dict = drift_report.as_dict()
    metrics_summary = result_dict["metrics"][0]["result"]
    number_of_features = metrics_summary["number_of_columns"]
    number_of_drifted_features = metrics_summary["number_of_drifted_columns"]
    
    drift_share = metrics_summary["share_of_drifted_columns"]
    print(f"📊 Statistical Covariate Drift: {number_of_drifted_features}/{number_of_features} features ({drift_share * 100:.1f}%)")

    # Save HTML artifact dashboard for manual administrative auditing
    os.makedirs("notebooks", exist_ok=True)
    drift_report.save_html("notebooks/data_drift_report.html")

    # 4. Conditional Gatekeeper Check
    if drift_share >= DRIFT_THRESHOLD:
        print(f"🚨 ALERT: Drift percentage has breached the safety metric threshold of {DRIFT_THRESHOLD*100:.0f}%!")
        print("🔪 Slicing drifted rows out of production logs for data labeling serialization...")
        
        drifted_export_df = current_df[['text']].copy()
        
        os.makedirs("data/datalake_landing", exist_ok=True)
        lake_path = "data/datalake_landing/drifted_raw_batch.csv"
        drifted_export_df.to_csv(lake_path, index=False)
        print(f"💾 Raw unlabelled data sliced and backed up to: {lake_path}")
        
        # Deploy Ollama localized multi-label grounding annotator task
        trigger_ollama_labeling_service(lake_path)
    else:
        print("✅ System Stable. Drift is within acceptable performance parameters.")

def trigger_ollama_labeling_service(lake_path):
    print("\n🦙 Initializing Ollama (Llama 3) Structured Multi-Label Annotation Workers...")
    raw_texts = pd.read_csv(lake_path)
    
    system_prompt = (
        "You are an expert AI content moderation system. Analyze the provided text and classify it "
        "across 5 categories. You must respond ONLY with a JSON object containing the keys: "
        "'label_hate_speech', 'label_insult', 'label_humiliate', 'label_dehumanize', 'label_violence'. "
        "The value for each key must strictly be either 0 (False) or 1 (True). "
        "Do not include any pleasantries, markdown blocks, or explanation text. Output pure JSON."
    )

    labeled_rows = []
    
    for i, row in enumerate(raw_texts.iterrows()):
        text = row[1]['text']
        print(f"   [Labeling {i+1}/{len(raw_texts)}] Text Sequence: \"{text[:40]}...\"")
        
        try:
            response = ollama.chat(
                model='llama3:latest',
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': f"Classify this text: {text}"}
                ],
                options={'temperature': 0.0, 'format': 'json'} # Forces structural JSON mapping outputs natively
            )
            
            annotations = json.loads(response['message']['content'])
            
            labeled_rows.append({
                "comment_id": i + 10000,
                "text": text,
                "label_hate_speech": int(annotations.get("label_hate_speech", 0)),
                "label_insult": int(annotations.get("label_insult", 0)),
                "label_humiliate": int(annotations.get("label_humiliate", 0)),
                "label_dehumanize": int(annotations.get("label_dehumanize", 0)),
                "label_violence": int(annotations.get("label_violence", 0))
            })
            
        except Exception as e:
            print(f"   ⚠️ Parsing Error on row {i}: {e}. Enforcing fallback zero-safety maps.")
            labeled_rows.append({
                "comment_id": i + 10000, "text": text,
                "label_hate_speech": 0, "label_insult": 0, "label_humiliate": 0, "label_dehumanize": 0, "label_violence": 0
            })

    # Export structured output mapping
    labeled_df = pd.DataFrame(labeled_rows)
    output_merge_path = "data/processed/new_production_labeled.csv"
    labeled_df.to_csv(output_merge_path, index=False)
    
    print(f"\n📥 LLM Ground-truth generation complete! Verified matrix schema written to: {output_merge_path}")
    print("🔄 Next step: Execute src/merge_and_retrain.py to update data lineage and commit v2 hashes.")

if __name__ == "__main__":
    check_drift_and_slice()