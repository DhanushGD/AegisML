FROM python:3.10-slim

WORKDIR /app

# Install native compilation dependencies and curl for health check validation
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
COPY src/ ./src/

RUN pip3 install --no-cache-dir -r requirements.txt

# Hugging Face Spaces REQUIRES port 7860 for custom Docker containers
EXPOSE 7860

# Add a native FastAPI health validation check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:7860/ || exit 1

# Launch the combined FastAPI + Gradio server via Uvicorn
ENTRYPOINT ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "7860"]