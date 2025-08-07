# ────────────────────────────────────────────────────────────────
#  Dockerfile  –  Build container for FastAPI + DuckDB + Prophet
#  (connects to external MS SQL Server & Ollama host)
# ────────────────────────────────────────────────────────────────

# ---------- 1 : base image -------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ---------- 2 : system packages --------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
         build-essential gcc g++ \
         curl gnupg ca-certificates wget \
         unixodbc-dev libssl-dev libpq-dev && \
    # --- Microsoft ODBC Driver 18 for SQL Server (Debian 12 “bookworm”) ---
    curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/debian/12/prod.list \
         -o /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql18 mssql-tools18 && \
    # tidy‑up
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------- 3 : Python deps ------------------------------------
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- 4 : copy source code -------------------------------
COPY main_duck.py ./main_duck.py

# (optional) create mount points for cache & models
RUN mkdir -p /app/parquet_cache /app/models

# ---------- 5 : runtime ----------------------------------------
EXPOSE 8000
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434  \
    PARQUET_TTL_H=24

CMD ["uvicorn", "main_duck:app", "--host", "0.0.0.0", "--port", "8000"]
