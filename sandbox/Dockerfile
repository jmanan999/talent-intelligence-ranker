FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY sandbox/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-cache embedding model (~130 MB) at build time
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5', cache_folder='./artifacts/model_cache')" 2>/dev/null || echo "Model pre-download skipped"

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

CMD ["streamlit", "run", "sandbox/app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
