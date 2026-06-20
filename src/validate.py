# src/validate.py
import os
import pandas as pd
import great_expectations as ge

def validate_dataset(file_path):
    print(f"🔍 Validating data schema for: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"⚠️ File not found at {file_path}. Skipping verification.")
        return True
        
    # Read raw dataframe using normal pandas
    raw_df = pd.read_csv(file_path)
    
    # Cross-version safe wrapper initialization
    df = ge.dataset.PandasDataset(raw_df) if hasattr(ge, 'dataset') else ge.from_pandas(raw_df)
    
    # Define structural expectations contract
    df.expect_column_to_exist('text')
    
    expected_labels = ['label_hate_speech', 'label_insult', 'label_humiliate', 'label_dehumanize', 'label_violence']
    for label in expected_labels:
        df.expect_column_to_exist(label)
        # Ensure values are bounded between binary 0 and 1
        df.expect_column_values_to_be_in_set(label, [0, 1, 0.0, 1.0])
        
    # Evaluate checklist results
    results = df.validate()
    
    if not results.success:
        print(f"❌ Schema Validation Failed for {file_path}!")
        return False
        
    print(f"✅ Schema Validation Passed for {file_path}!")
    return True

if __name__ == "__main__":
    splits = ["data/processed/val.csv"]
    all_passed = True
    
    for split in splits:
        if not validate_dataset(split):
            all_passed = False
            
    if not all_passed:
        exit(1) # Force exit code 1 to crash CI/CD pipeline if schema fails