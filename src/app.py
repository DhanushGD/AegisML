# src/app.py
import os
import sqlite3
import torch
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Define the expected JSON input structure
class CommentRequest(BaseModel):
    text: str

# Define the structured output format
class PredictionResponse(BaseModel):
    text: str
    predictions: dict

# Initialize FastAPI App
app = FastAPI(
    title="Real-Time Toxic Comment Classifier API",
    description="Production-ready REST API for multi-label text moderation pulling from Hugging Face Hub."
)

# Global variables for model, tokenizer, and configuration
# ==========================================================
# 🌐 PRODUCTION CONFIG: Points straight to your HF Hub Model Registry!
# ==========================================================
MODEL_REPO_ID = "Luffyyyyyyyyyyyyyyyyy/toxic-distilbert" 
DB_PATH = "data/production_logs.db"
TARGET_LABELS = ['label_hate_speech', 'label_insult', 'label_humiliate', 'label_dehumanize', 'label_violence']

tokenizer = None
model = None

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

@app.on_event("startup")
def startup_event():
    global tokenizer, model
    
    # Initialize our database logging framework
    init_db()
    
    print(f"📥 Downloading and caching weights directly from HF Hub: {MODEL_REPO_ID}...")
    try:
        # HuggingFace automatically downloads, caches, and streams the weights directly from your account
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_REPO_ID)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_REPO_ID)
        model.eval() 
        print("🚀 API Server is online and authenticated with Hugging Face Hub registry!")
    except Exception as e:
        raise RuntimeError(f"Failed to load model from Hugging Face Hub: {str(e)}")

@app.get("/")
def health_check():
    return {"status": "healthy", "model_source": MODEL_REPO_ID}

@app.post("/predict", response_model=PredictionResponse)
def predict_toxicity(payload: CommentRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text payload cannot be empty.")
        
    try:
        # Tokenize incoming raw string
        inputs = tokenizer(
            payload.text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        )
        
        # Disable gradient calculations for production speed
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.squeeze().numpy()
            
        # Apply Sigmoid activation function to get continuous probability metrics
        probabilities = 1 / (1 + np.exp(-logits))
        
        # Create user response dictionary
        prediction_map = {
            TARGET_LABELS[i]: float(probabilities[i]) 
            for i in range(len(TARGET_LABELS))
        }
        
        # ==========================================================
        # 💾 AUDIT TRAIL LOGGING: Log live inferences to SQLite 
        # ==========================================================
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO inference_logs (
                timestamp, input_text, label_hate_speech, label_insult, label_humiliate, label_dehumanize, label_violence
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.utcnow().isoformat(),
            payload.text,
            prediction_map['label_hate_speech'],
            prediction_map['label_insult'],
            prediction_map['label_humiliate'],
            prediction_map['label_dehumanize'],
            prediction_map['label_violence']
        ))
        conn.commit()
        conn.close()
        
        return PredictionResponse(text=payload.text, predictions=prediction_map)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine Error: {str(e)}")