# ===== Stage 1: Build React frontend =====
FROM node:20-slim AS frontend-build

WORKDIR /frontend

# Install dependencies (--legacy-peer-deps for shadcn/radix compatibility)
# Only copy package.json — skip lockfile since it was generated on Windows
# and lacks Linux-native bindings for Rollup/Vite
COPY frontend/package.json ./
RUN npm install --legacy-peer-deps

# Build the production bundle
COPY frontend/ .
RUN npm run build

# ===== Stage 2: Python backend with FastAPI + sentence-transformers =====
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (for transformers/torch compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy backend code and install Python dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/

# Copy the 30 ClearPath PDFs
COPY docs/ ./docs/

# Copy built React assets from Stage 1
COPY --from=frontend-build /frontend/dist ./frontend_build

# Pre-download the sentence-transformers model at build time
# to avoid cold-start latency on first request
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Backend imports use 'from app.xxx' so backend/ must be on PYTHONPATH
ENV PYTHONPATH=/app/backend

# Cloud Run expects the app to listen on $PORT (default 8080)
ENV PORT=8080

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
