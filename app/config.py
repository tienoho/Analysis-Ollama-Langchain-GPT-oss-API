import os
from pathlib import Path
from sqlalchemy import create_engine

SQL_URI = os.getenv(
    "SQL_URI",
    "mssql+pyodbc://USER:PASSWORD@SERVER:1433/ERPDB"
    "?driver=ODBC+Driver+17+for+SQL+Server",
)
PARQUET_DIR = Path("./parquet_cache")
MODEL_DIR = Path("./models")
PARQUET_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
PARQUET_TTL_H = int(os.getenv("PARQUET_TTL_H", 24))
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_NAME = "gpt-oss:20b"

engine = create_engine(SQL_URI, pool_pre_ping=True, future=True)
