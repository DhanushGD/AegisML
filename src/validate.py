# src/validate.py
import pandas as pd
import great_expectations as ge
from great_expectations.dataset.pandas_dataset import PandasDataset
import sys

def validate_dataset(file_path):
    print(f"🔍 Validating data schema for: {file_path}")
    raw_df = pd.read_csv(file_path)
    df = PandasDataset(raw_df)
    
    # Load data as a Great Expectations DataFrame
    # df = ge.from_pandas(pd.read_csv(file_path))
    
    # Define the updated multi-label columns we created in ingest.py
    label_columns = [
        'label_hate_speech', 
        'label_insult', 
        'label_humiliate', 
        'label_dehumanize', 
        'label_violence'
    ]
    
    # Rule 1: Columns must exist exactly as expected (Updated for Multi-Label schema)
    expected_columns = ['comment_id', 'text'] + label_columns
    res1 = df.expect_table_columns_to_match_ordered_list(expected_columns)
    
    # Rule 2: The text column must never contain null values
    res2 = df.expect_column_values_to_not_be_null('text')
    
    # Rule 3: Text must actually be string types
    res3 = df.expect_column_values_to_be_of_type('text', 'str')
    
    # Rule 4: Dynamically validate that every multi-label column contains ONLY 0 or 1
    label_checks = {}
    all_labels_valid = True
    for col in label_columns:
        res_label = df.expect_column_values_to_be_in_set(col, [0, 1])
        label_checks[col] = res_label
        if not res_label.success:
            all_labels_valid = False

    # Check overall status
    if res1.success and res2.success and res3.success and all_labels_valid:
        print(f"✅ {file_path} passed all multi-label data quality checks!")
        return True
    else:
        print(f"❌ {file_path} FAILED data quality checks!")
        # Print breakdown of failures for easier debugging
        for name, res in [("Columns match", res1), ("Text not null", res2), ("Text is string", res3)]:
            if not res.success:
                print(f"   -> Failed check: {name}")
                
        for col, res in label_checks.items():
            if not res.success:
                print(f"   -> Failed check: {col} column is not purely binary (0 or 1)")
        return False

if __name__ == "__main__":
    # Validate all three splits
    splits = ["data/processed/train.csv", "data/processed/val.csv", "data/processed/test.csv"]
    
    all_passed = True
    for split in splits:
        if not validate_dataset(split):
            all_passed = False
            
    if not all_passed:
        print("🚨 Data validation failed. Stopping pipeline.")
        sys.exit(1)
    else:
        print("🚀 All multi-label data splits are clean and verified!")