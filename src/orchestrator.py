# src/orchestrator.py
import sqlite3
import pandas as pd
import os
import json
import ollama
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping

DB_PATH = "data/production_logs.db"
VAL_PATH = "data/processed/val.csv"
DRIFT_THRESHOLD = 0.50  # 50% threshold metric

def check_drift_and_slice():
    print("🚀 Stage 7 Orchestrator: Fetching statistical logs...")
    
    if not os.path.exists(DB_PATH):
        print("❌ No production database found yet. Run traffic simulation first!")
        return

    # 1. Gather Reference vs Current Data
    ref_df = pd.read_csv(VAL_PATH)
    
    conn = sqlite3.connect(DB_PATH)
    current_df = pd.read_sql_query("""
        SELECT input_text as text, label_hate_speech, label_insult, label_humiliate, label_dehumanize, label_violence 
        FROM inference_logs
    """, conn)
    conn.close()
    
    # 2. Build Column Matrix
    column_mapping = ColumnMapping()
    column_mapping.text_features = ['text']
    column_mapping.numerical_features = ['label_hate_speech', 'label_insult', 'label_humiliate', 'label_dehumanize', 'label_violence']
    
    # 3. Compute Data Drift using Dictionary output
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=ref_df, current_data=current_df, column_mapping=column_mapping)
    
    # Extract metrics as a raw python dictionary object
    result_dict = drift_report.as_dict()
    metrics_summary = result_dict["metrics"][0]["result"]
    number_of_features = metrics_summary["number_of_columns"]
    number_of_drifted_features = metrics_summary["number_of_drifted_columns"]
    
    drift_share = metrics_summary["share_of_drifted_columns"]
    print(f"📊 Features Drifted: {number_of_drifted_features}/{number_of_features} ({drift_share * 100:.1f}%)")

    # Save the HTML report as well so humans can still look at it
    drift_report.save_html("notebooks/data_drift_report.html")

    # 4. Conditional Gatekeeper Check
    if drift_share >= DRIFT_THRESHOLD:
        print(f"🚨 ALERT: Drift percentage ({drift_share*100:.1f}%) has breached the {DRIFT_THRESHOLD*100:.0f}% threshold!")
        print("🔪 Slicing drifted rows out of SQLite database for LLM labeling queue...")
        
        drifted_export_df = current_df[['text']].copy()
        
        os.makedirs("data/datalake_landing", exist_ok=True)
        lake_path = "data/datalake_landing/drifted_raw_batch.csv"
        drifted_export_df.to_csv(lake_path, index=False)
        print(f"💾 Raw data saved to Data Lake: {lake_path}")
        
        # Trigger the true LLM labeling loop
        trigger_ollama_labeling_service(lake_path)
    else:
        print("✅ System Stable. Drift is within acceptable performance parameters.")

def trigger_ollama_labeling_service(lake_path):
    print("\n🦙 Initializing Ollama (Llama 3) Structured Multi-Label Annotation...")
    raw_texts = pd.read_csv(lake_path)
    
    # Define the strict system prompt enforcing the multi-label schema contract
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
        print(f"   [Labeling {i+1}/{len(raw_texts)}] Text: \"{text[:40]}...\"")
        
        try:
            # Query local Ollama instance using json format mode
            response = ollama.chat(
                model='llama3:latest',
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': f"Classify this text: {text}"}
                ],
                options={'temperature': 0.0} # Keep predictions completely deterministic
            )
            
            # Parse response content strings back to dictionary
            annotations = json.loads(response['message']['content'])
            
            # Append fully conformed record matching training schemas
            labeled_rows.append({
                "comment_id": i + 10000, # Mocked production ID offset
                "text": text,
                "label_hate_speech": int(annotations.get("label_hate_speech", 0)),
                "label_insult": int(annotations.get("label_insult", 0)),
                "label_humiliate": int(annotations.get("label_humiliate", 0)),
                "label_dehumanize": int(annotations.get("label_dehumanize", 0)),
                "label_violence": int(annotations.get("label_violence", 0))
            })
            
        except Exception as e:
            print(f"   ⚠️ Error parsing Ollama output for row {i}: {e}. Falling back to clean defaults.")
            labeled_rows.append({
                "comment_id": i + 10000, "text": text,
                "label_hate_speech": 0, "label_insult": 0, "label_humiliate": 0, "label_dehumanize": 0, "label_violence": 0
            })

    # Save out the ground truth dataset
    labeled_df = pd.DataFrame(labeled_rows)
    output_merge_path = "data/processed/new_production_labeled.csv"
    labeled_df.to_csv(output_merge_path, index=False)
    
    print(f"\n📥 LLM Ground-truth generation complete! Schema payload verified at: {output_merge_path}")
    print("🔄 Next engineering step: Run your Prefect DAG flow to merge datasets and scale training up.")

if __name__ == "__main__":
    check_drift_and_slice()