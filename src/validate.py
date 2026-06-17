# src/validate.py
import pandas as pd
import great_expectations as ge
import sys

def validate_dataset(file_path):
    print(f"🔍 Validating data schema for: {file_path}")
    
    # Load data as a Great Expectations DataFrame
    df = ge.from_pandas(pd.read_csv(file_path))
    
    # Rule 1: Columns must exist exactly as expected
    expected_columns = ['comment_id', 'text', 'hatespeech', 'insult', 'dehumanize', 'violence', 'genocide']
    res1 = df.expect_table_columns_to_match_ordered_list(expected_columns)
    
    # Rule 2: The text column must never contain null values
    res2 = df.expect_column_values_to_not_be_null('text')
    
    # Rule 3: Text must actually be string types
    res3 = df.expect_column_values_to_be_of_type('text', 'str')
    
    # Rule 4: Label columns must contain only 0 or 1
    target_cols = ['hatespeech', 'insult', 'dehumanize', 'violence', 'genocide']
    label_success = True
    for col in target_cols:
        res_col = df.expect_column_values_to_be_in_set(col, [0, 1])
        if not res_col.success:
            label_success = False

    # Check overall status
    if res1.success and res2.success and res3.success and label_success:
        print(f"✅ {file_path} passed all data quality checks!")
        return True
    else:
        print(f"❌ {file_path} FAILED data quality checks!")
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
        print("🚀 All data splits are clean and verified!")