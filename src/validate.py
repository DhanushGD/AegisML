# src/validate.py
import pandas as pd
import great_expectations as ge
import sys

def validate_dataset(file_path):
    print(f"🔍 Validating data schema for: {file_path}")
    
    # Load data as a Great Expectations DataFrame
    df = ge.from_pandas(pd.read_csv(file_path))
    
    # Rule 1: Columns must exist exactly as expected (Updated for IRT-based schema)
    expected_columns = ['comment_id', 'text', 'is_hate_speech']
    res1 = df.expect_table_columns_to_match_ordered_list(expected_columns)
    
    # Rule 2: The text column must never contain null values
    res2 = df.expect_column_values_to_not_be_null('text')
    
    # Rule 3: Text must actually be string types
    res3 = df.expect_column_values_to_be_of_type('text', 'str')
    
    # Rule 4: The new binary label column must contain only 0 or 1
    res4 = df.expect_column_values_to_be_in_set('is_hate_speech', [0, 1])

    # Check overall status
    if res1.success and res2.success and res3.success and res4.success:
        print(f"✅ {file_path} passed all data quality checks!")
        return True
    else:
        print(f"❌ {file_path} FAILED data quality checks!")
        # Print breakdown of failures for easier debugging
        for name, res in [("Columns match", res1), ("Text not null", res2), ("Text is string", res3), ("Label binary", res4)]:
            if not res.success:
                print(f"   -> Failed check: {name}")
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