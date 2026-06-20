# src/app.py
import os
import sqlite3
import torch
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import gradio as gr
# Define the expected JSON input structure
class CommentRequest(BaseModel):
    text: str

# Define the structured output format
class PredictionResponse(BaseModel):
    text: str
    predictions: dict



# Global variables for model, tokenizer, and configuration
# ==========================================================
# 🌐 PRODUCTION CONFIG: Points straight to your HF Hub Model Registry!
# ==========================================================
MODEL_REPO_ID = "Luffyyyyyyyyyyyyyyyyy/toxic-distilbert" 
DB_PATH = "data/production_logs.db"
TARGET_LABELS = ['label_hate_speech', 'label_insult', 'label_humiliate', 'label_dehumanize', 'label_violence']

tokenizer = None
model = None

from contextlib import asynccontextmanager

def init_db():
    """Creates a local SQLite database to track incoming inferences for Stage 5 monitoring."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
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
    ''')
    conn.commit()
    conn.close()
    print(f"🗄️ SQLite database initialized at {DB_PATH}")


# Shared inference logic used by both FastAPI and Gradio UI
def run_inference(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze().detach().cpu().numpy()
    probabilities = 1 / (1 + np.exp(-logits))
    
    prediction_map = {TARGET_LABELS[i]: float(probabilities[i]) for i in range(len(TARGET_LABELS))}
    
    # Log prediction to SQLite database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO inference_logs (timestamp, input_text, label_hate_speech, label_insult, label_humiliate, label_dehumanize, label_violence)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (datetime.utcnow().isoformat(), text, *[prediction_map[l] for l in TARGET_LABELS]))
    conn.commit()
    conn.close()
    
    return prediction_map

# Modern Lifespan Handler for Asset Preloading
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model
    # Run setup operations
    init_db()
    print(f"📥 Downloading and caching weights directly from HF Hub: {MODEL_REPO_ID}...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_REPO_ID)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_REPO_ID)
    model.eval()
    print("🚀 API Server is online and authenticated with Hugging Face Hub registry!")
    yield

# Initialize FastAPI App
app = FastAPI(
    title="Real-Time Toxic Comment Classifier API",
    description="Production-ready REST API for multi-label text moderation pulling from Hugging Face Hub.",
    lifespan=lifespan
)

@app.get("/")
def health_check():
    return {"status": "healthy", "model_source": MODEL_REPO_ID}

@app.post("/predict")
def predict_endpoint(payload: CommentRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    return {"text": payload.text, "predictions": run_inference(payload.text)}

from fastapi.responses import FileResponse

# Add this endpoint inside src/app.py
@app.get("/download_logs")
def download_logs():
    if os.path.exists(DB_PATH):
        return FileResponse(DB_PATH, filename="production_logs.db", media_type="application/x-sqlite3")
    raise HTTPException(status_code=404, detail="No database records found yet.")

# --- GRADIO INTERACTIVE UI SETUP ---
def gradio_interface(text):
    if not text.strip():
        return "Please enter valid text."
    results = run_inference(text)
    # Format the dictionary output nicely for the UI screen
    return {label.replace("label_", "").upper(): score for label, score in results.items()}

demo = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(lines=2, placeholder="Type a comment to evaluate toxicity..."),
    outputs=gr.Label(num_top_classes=5),
    title="🛡️ AI Real-Time Content Moderation Gatekeeper",
    description="This multi-label classifier scores text semantics instantly using fine-tuned DistilBERT models."
)

# Mount Gradio UI directly onto the root URL of your FastAPI application
app = gr.mount_gradio_app(app, demo, path="/")