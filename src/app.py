# src/app.py
print("APP.PY LOADED")
import os
import sqlite3
from datetime import datetime
from contextlib import asynccontextmanager

import gradio as gr
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)

print("APP STARTING...")

# ==========================================================
# CONFIG
# ==========================================================
MODEL_REPO_ID = "Luffyyyyyyyyyyyyyyyyy/toxic-distilbert"
DB_PATH = "data/production_logs.db"

TARGET_LABELS = [
    "label_hate_speech",
    "label_insult",
    "label_humiliate",
    "label_dehumanize",
    "label_violence",
]

tokenizer = None
model = None

# ==========================================================
# REQUEST / RESPONSE MODELS
# ==========================================================
class CommentRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    predictions: dict

# ==========================================================
# DATABASE
# ==========================================================
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS inference_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            input_text TEXT,
            label_hate_speech REAL,
            label_insult REAL,
            label_humiliate REAL,
            label_dehumanize REAL,
            label_violence REAL
        )
        """
    )
    conn.commit()
    conn.close()
    print(f"[OK] SQLite initialized: {DB_PATH}")

print("Initializing database...")
init_db()

print("Loading tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_REPO_ID)

print("Loading model...")
model = DistilBertForSequenceClassification.from_pretrained(MODEL_REPO_ID)
model.eval()
print("MODEL LOADED SUCCESSFULLY")

# ==========================================================
# FASTAPI INSTANCE
# ==========================================================
app = FastAPI(
    title="Real-Time Toxic Comment Classifier API",
    description="Production-ready REST API with integrated Gradio UI",
)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_source": MODEL_REPO_ID,
    }

# ==========================================================
# INFERENCE CORE
# ==========================================================
def run_inference(text: str):
    if tokenizer is None or model is None:
        raise RuntimeError("Model not loaded.")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits.squeeze().detach().cpu().numpy()
    probabilities = 1 / (1 + np.exp(-logits))

    prediction_map = {
        TARGET_LABELS[i]: float(probabilities[i])
        for i in range(len(TARGET_LABELS))
    }

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO inference_logs
        (timestamp, input_text, label_hate_speech, label_insult, label_humiliate, label_dehumanize, label_violence)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(),
            text,
            *[prediction_map[l] for l in TARGET_LABELS],
        ),
    )
    conn.commit()
    conn.close()

    return prediction_map

@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(payload: CommentRequest):
    if not payload.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty.",
        )
    return {
        "text": payload.text,
        "predictions": run_inference(payload.text),
    }

@app.get("/download_logs")
def download_logs():
    if os.path.exists(DB_PATH):
        return FileResponse(
            DB_PATH,
            filename="production_logs.db",
            media_type="application/x-sqlite3",
        )
    raise HTTPException(
        status_code=404,
        detail="No database records found yet.",
    )

# ==========================================================
# GRADIO INTERFACE
# ==========================================================
def gradio_predict(text):
    if not text.strip():
        return {"ERROR_EMPTY_TEXT": 1.0}
    results = run_inference(text)
    return {
        label.replace("label_", "").upper(): score
        for label, score in results.items()
    }

with gr.Blocks() as demo:
    gr.Markdown("# 🛡️ AI Toxic Comment Classifier")
    gr.Markdown("Multi-label DistilBERT moderation system with exposed REST API endpoints.")
    
    text_input = gr.Textbox(lines=3, placeholder="Type a comment to moderate...")
    output = gr.Label()
    predict_btn = gr.Button("Evaluate Content")
    
    predict_btn.click(
        fn=gradio_predict,
        inputs=text_input,
        outputs=output,
    )

print("GRADIO BLOCKS CREATED")

# 💥 FIX: Mount Gradio on "/" so the public browser layout renders immediately 
# Your FastAPI endpoints remain fully live right alongside it on /predict and /download_logs!
app = gr.mount_gradio_app(app, demo, path="/")

print("APPLICATION READY FOR DEPLOYMENT")