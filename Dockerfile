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

# ===== Stage 2: Python backend with FastAPI + ONNX Runtime =====
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY docs/ ./docs/
COPY --from=frontend-build /frontend/dist ./frontend_build

# ONNX model is committed to backend/onnx_model/ (~90MB)
# No need to download at build time — already in the repo

ENV PYTHONPATH=/app/backend
ENV PORT=8080

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
