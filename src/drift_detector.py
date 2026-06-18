# src/drift_detector.py

import sqlite3
import traceback

import pandas as pd

from evidently.report import Report
from evidently.metric_preset import TextOverviewPreset
from evidently import ColumnMapping


def run_drift_analysis():
    print("🕵️ Running Monitoring Engine via Evidently AI...")

    try:
        # =====================================================
        # 1. LOAD REFERENCE DATA
        # =====================================================
        print("\n[STEP 1] Loading reference dataset...")

        ref_df = pd.read_csv("data/processed/val.csv")

        print(f"Reference shape: {ref_df.shape}")
        print(f"Reference columns: {list(ref_df.columns)}")

        if "text" not in ref_df.columns:
            raise ValueError(
                f"'text' column not found in validation data. Available columns: {list(ref_df.columns)}"
            )

        ref_text = ref_df[["text"]].copy()

        # =====================================================
        # 2. LOAD CURRENT DATA
        # =====================================================
        print("\n[STEP 2] Loading production logs...")

        conn = sqlite3.connect("data/production_logs.db")

        current_df = pd.read_sql_query(
            """
            SELECT input_text AS text
            FROM inference_logs
            """,
            conn,
        )

        conn.close()

        print(f"Current shape: {current_df.shape}")
        print(f"Current columns: {list(current_df.columns)}")

        if len(current_df) < 5:
            print(
                f"⚠️ Not enough production logs for drift analysis ({len(current_df)} rows)"
            )
            return

        # =====================================================
        # 3. DATA QUALITY CHECKS
        # =====================================================
        print("\n[STEP 3] Data validation...")

        print("\nReference dtypes:")
        print(ref_text.dtypes)

        print("\nCurrent dtypes:")
        print(current_df.dtypes)

        print("\nReference null counts:")
        print(ref_text.isnull().sum())

        print("\nCurrent null counts:")
        print(current_df.isnull().sum())

        # Force text columns to strings
        ref_text["text"] = ref_text["text"].fillna("").astype(str)
        current_df["text"] = current_df["text"].fillna("").astype(str)

        print("\nReference dtype AFTER conversion:")
        print(ref_text["text"].dtype)

        print("\nCurrent dtype AFTER conversion:")
        print(current_df["text"].dtype)

        print("\nReference empty strings:")
        print((ref_text["text"].str.strip() == "").sum())

        print("\nCurrent empty strings:")
        print((current_df["text"].str.strip() == "").sum())

        print("\nReference sample:")
        print(ref_text.head())

        print("\nCurrent sample:")
        print(current_df.head())

        print(
            f"\nComparing {len(ref_text)} reference samples vs {len(current_df)} production samples"
        )

        # =====================================================
        # 4. COLUMN MAPPING
        # =====================================================
        print("\n[STEP 4] Configuring column mapping...")

        column_mapping = ColumnMapping()
        column_mapping.text_features = ["text"]

        print("Text features:", column_mapping.text_features)

        # =====================================================
        # 5. BUILD REPORT
        # =====================================================
        print("\n[STEP 5] Creating Evidently report...")

        text_report = Report(
            metrics=[
                TextOverviewPreset(column_name="text")
            ]
        )

        # =====================================================
        # 6. RUN REPORT
        # =====================================================
        print("\n[STEP 6] Running report...")

        text_report.run(
            reference_data=ref_text,
            current_data=current_df,
            column_mapping=column_mapping,
        )

        print("✅ Report execution completed")

        # =====================================================
        # 7. SAVE HTML
        # =====================================================
        print("\n[STEP 7] Saving HTML dashboard...")

        output_html = "notebooks/data_drift_report.html"

        text_report.save_html(output_html)

        print(f"📊 Monitoring complete!")
        print(f"Dashboard saved to: {output_html}")

    except Exception as e:
        print("\n❌ DRIFT ANALYSIS FAILED")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")

        print("\nFull traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    run_drift_analysis()