# Step 1.1: Initialize Version Control (Git + DVC)
We want Git to track your code, but we absolutely do not want it tracking heavy CSV files. That is where Data Version Control (DVC) comes in.
python src/ingest.py
# Initialize Git
git init
<!-- $env:Path += ";C:\Program Files\Git\bin;C:\Program Files\Git\cmd" -->

# Initialize DVC
dvc init

# Configure a local or remote storage directory for your data (e.g., a local cache folder simulating a cloud bucket)
mkdir -p /tmp/dvc_remote
dvc remote add -d myremote /tmp/dvc_remote

# Track the Output with DVC:
dvc add data/processed
This commands DVC to take a hash snapshot of the folder. It creates a lightweight pointer file named data/processed.dvc and updates your .gitignore so Git ignores the actual large data folder.
# Save data snapshot to DVC local storage
dvc push
# Track the pointer files and code in Git
git add data/processed.dvc .gitignore src/ingest.py config/ requirements.txt
git commit -m "feat: complete step 1 data infrastructure baseline layout"

# Your Milestone Checklist
Before moving to Step 2 (Training Core), ensure you can successfully run this reproducibility check in your terminal:
Run rm -rf data/processed/ (Simulate a clean machine or a teammate cloning your repo).
Run dvc pull.
Verify that your data/processed/ directory instantly reappeared with your exact CSV splits intact.

# Data Validation with Great Expectations
We will write a automated script that runs validation rules on your train.csv, val.csv, and test.csv to ensure they match our architectural contract.

python src/validate.py

git add src/validate.py
git commit -m "feat: add data validation layer with great expectations"