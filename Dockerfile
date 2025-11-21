FROM python:3.10-slim

# -----------------------------
# 1. System dependencies
# -----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# 2. Create app working dir
# -----------------------------
WORKDIR /app

# -----------------------------
# 3. Copy only requirements first (better caching)
# -----------------------------
COPY requirements.txt .

# -----------------------------
# 4. Install Python packages
# -----------------------------
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# 5. Copy full project
# -----------------------------
COPY . .

# -----------------------------
# 6. Environment variables
# -----------------------------
# Replace with your HF token if needed (Render automatically injects secrets)
ENV HF_HOME=/app/hf_cache \
    TRANSFORMERS_CACHE=/app/hf_cache

# -----------------------------
# 7. Expose port + start server
# -----------------------------
EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
