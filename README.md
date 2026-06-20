---
title: Toxic Classifier API
emoji: 🛡️
colorFrom: red
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - fastapi
  - gradio
  - transformers
  - mlops
pinned: false
short_description: Asynchronous Multi-Label Toxicity Moderation API Gateway
license: mit
---

# 🛡️ Real-Time Toxic Comment Classifier MLOps Infrastructure

An end-to-end, self-healing production infrastructure built around a fine-tuned sequence classification model (**DistilBERT**). This system moves beyond simple model training by implementing live data schemas, robust model observability, automated drift tracking, and an intentional **Human-in-the-Loop (HITL)** re-labeling loop using a local LLM to drive continuous training workflows safely and deterministically.

---

## 🏗️ Core System Architecture

The pipeline functions as a closed-loop data lifecycle engine divided into distinct micro-layers:

```text
[ Live User Traffic ] ──► [ Gradio UI / FastAPI (HF Spaces) ] ──► [ Cloud Logs (SQLite) ]
                                                                        │
                                                                 (HITL DB Sync)
                                                                        ▼
[ Dataset v2 (DVC) ] ◄── [ Great Expectations ] ◄── [ Local Ollama (Llama 3) Labeler ]
        │
  (Git Push)
        ▼
[ GitHub Actions CI/CD ] ──► [ PyTest Suite Runs ] ──► [ Auto-Deploy to HF Spaces ]
```

1. **Production Edge**: User strings flow directly into a public FastAPI + Gradio container hosted on Hugging Face Spaces via Docker.
2. **Audit Trails**: Predictions are computed across 5 targets (hate_speech, insult, humiliate, dehumanize, violence) and appended into a production SQLite engine.
3. **Human-in-the-Loop (HITL) Interface**: The engineering operator runs the localized orchestrator to pull down production snapshots and run drift profiling.
4. **Autonomous Adaptation Loop**: If drift breaches 50%, a local Ollama (Llama 3) node automatically orchestrates multi-label grounding annotations to structure a fresh training matrix (Dataset v2).
5. **Continuous Deployment Firewall**: GitHub Actions asserts data quality schemas and test criteria before compiling zero-downtime microservice updates.

##  🛠️ The MLOps Stack
- **Model Core**: Hugging Face transformers (Fine-tuned DistilBERT)
- **API Delivery & Interface**: FastAPI + Uvicorn + Gradio App Mounting
- **Container Layer**: Docker (python:3.10-slim runtime base)
- **Data Observability**: Evidently AI (Data Drift Preset & Metric Tracking)\
- **Local Annotation Worker**: Ollama API (llama3:latest native JSON Mode)
- **Data Lineage**: Data Version Control (DVC)
- **Data Firewalls**: Great Expectations (Data Asset Validation Checklists)             
- **CI/CD Automati**on: GitHub Actions Pipelines (pytest + Cloud Registry Sync)

## 🧑‍💻 File Structure Mapping
```text
├── .github/workflows/
│   └── ci_cd.yml               # Automated GitHub Actions validation & deployment
├── data/
│   ├── processed/
│   │   ├── train.csv           # Core training historical matrices
│   │   └── val.csv             # Fixed reference dataset anchor split
│   └── datalake_landing/       # Storage tier for isolated drifted rows
├── notebooks/
│   └── data_drift_report.html  # Human-readable Evidently AI distribution graph
├── src/
│   ├── app.py                  # Dual-engine API (FastAPI REST + Gradio Frontend)
│   ├── orchestrator.py         # Main pipeline brain (HITL sync, volume check, Llama 3 loop)
│   ├── merge_and_retrain.py    # Merges V2 training data, calls DVC, and flushes DB
│   └── validate.py             # Great Expectations structural schema firewall
├── Dockerfile                  # Uvicorn container orchestration parameters
├── requirements.txt            # Explicit project dependency limits
└── README.md                   # System configuration and blueprint
```

## 🔄 The Human-In-The-Loop (HITL) Data Pipeline Flow
To prevent training drift and avoid wasting expensive GPU compute budgets on tiny, unverified data increments, this infrastructure enforces a structured, batched data-collection loop:

**1. High-Volume Log Gathering**
Live end-user inputs from your public space populate the database endpoint (/download_logs).

**2. The Manual Orchestration Sync (HITL Action)**
As an administrative operator, you run the orchestrator tool from your workspace terminal:
```bash
python src/orchestrator.py
```
- **Volume Gate**: The script triggers an HTTP request, downloads the live logs, and counts the rows. If it is less than 1,000 entries, the script safely exits to protect downstream resource costs.
- **Drift Profiler**: If data exceeds 1,000 rows, Evidently AI evaluates vocabulary shift signatures. If embedding distribution drift crosses your 50% barrier, the orchestrator ringfences the drifted texts.

**3. Automated Ground-Truth Labeling & Verification**
The orchestrator awakens your local Ollama Llama 3 engine. It streams the drifted rows through a strict system prompt instruction, enforcing a deterministic multi-label JSON serialization contract. This fixes schema mismatch issues by cleanly formatting data back to original modeling dimensions:
```bash
{
  "label_hate_speech": 0, "label_insult": 1, "label_humiliate": 0, "label_dehumanize": 0, "label_violence": 1
}
```
**4. Data Lineage Merging & DB Table Flush**
You run the secondary processing script:
```bash
python src/merge_and_retrain.py
```
This script blends the fresh annotations into train.csv, commits the new hashes inside Data Version Control (DVC), and runs an atomic DELETE query to clear out the production table logs, resetting your 1,000-row tracking metric gate for the next cycle.

## 🚀 Local Workspace Setup & Installation
**1. Clone the Codebase and Prepare Environments**
```bash
git clone [https://github.com/DhanushGD/toxic-classifier.git](https://github.com/DhanushGD/toxic-classifier.git)
cd toxic-classifier
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

**2. Ensure Local Dependencies are Active**
Download and open the Ollama desktop client, then pull the core model weights model in your terminal:
```bash
ollama pull llama3:latest
```

**3. Run Automated Local Unit Tests**
```bash
pytest src/test_api.py
```

## 🛡️ Cloud Automated CI/CD Deployment
The repository is fully protected via an automated **GitHub Actions** workflow engine (ci_cd.yml):
- Data Quality Firewall: Every code push or pull request automatically invokes Great Expectations to test input columns and binary ranges.
- Regression Safety Check: The engine boots a headless API instance and fires integration scripts (pytest) to assert route status integrity.
- Continuous Deployment Sync: Once code evaluations pass green, GitHub Actions securely retrieves your HF_TOKEN credentials, overrides target container registries, and updates your live Hugging Face Space app layout instantly with zero manual production downtime!