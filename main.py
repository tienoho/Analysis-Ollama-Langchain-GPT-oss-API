"""
main_duck.py  ───────────────────────────────────────────────────────────
✓  MS SQL Server → Parquet cache  → DuckDB aggregate
✓  Prophet / ARIMA forecast
✓  Optional GPT‑oss (Ollama) narrative
─────────────────────────────────────────────────────────────────────────
PIP DEPS
pip install fastapi uvicorn[standard] pandas numpy sqlalchemy pyodbc \
            duckdb prophet pmdarima langchain langchain-community \
            langchain-ollama python-dotenv
"""

from __future__ import annotations

import os, hashlib, pickle, datetime as dt
from pathlib import Path
from typing import Literal, List

import pandas as pd
import duckdb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, conint
from sqlalchemy import create_engine, text
from prophet import Prophet
import pmdarima as pm

# LangChain (LLM narrative)
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

# ───────────────────────────────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────────────────────────────

SQL_URI = os.getenv(
    "SQL_URI",
    "mssql+pyodbc://USER:PASSWORD@SERVER:1433/ERPDB"
    "?driver=ODBC+Driver+17+for+SQL+Server",
)
PARQUET_DIR = Path("./parquet_cache")
MODEL_DIR = Path("./models")
PARQUET_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

PARQUET_TTL_H = int(os.getenv("PARQUET_TTL_H", 24))  # làm mới sau 24 giờ

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_NAME = "gpt-oss:20b"

engine = create_engine(SQL_URI, pool_pre_ping=True, future=True)

# ───────────────────────────────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMA
# ───────────────────────────────────────────────────────────────────────

class Req(BaseModel):
    metric: Literal["revenue", "production_output", "import_volume"]
    frequency: Literal["M", "Q", "Y"] = "M"
    horizon: conint(ge=6, le=12) = 12
    model: Literal["prophet", "arima", "auto"] = "auto"
    use_llm: bool = False


class Pred(BaseModel):
    period: str
    forecast: float
    lower: float
    upper: float


class Res(BaseModel):
    metric: str
    frequency: str
    horizon: int
    unit: Literal["USD", "units"]
    model_used: str
    predictions: List[Pred]
    narrative: str | None = None


app = FastAPI(title="DuckDB‑Parquet Forecast API")

# ───────────────────────────────────────────────────────────────────────
# UTILITIES – SQL + PARQUET
# ───────────────────────────────────────────────────────────────────────

def sql_for(metric: str) -> tuple[str, str]:
    """Return raw‑level SQL for metric + unit."""
    if metric == "revenue":
        sql = """
        SELECT CAST(okr.OrderDat AS DATE) AS ds,
               SUM(osg.Extra_pr)          AS y
        FROM   Orsrg osg
        JOIN   Orkrg okr ON okr.OrderNr = osg.OrderNr
        WHERE  okr.Or_Soort = 'V'         -- sales orders
        GROUP  BY CAST(okr.OrderDat AS DATE)
        """
        unit = "USD"
    elif metric == "production_output":
        sql = """
        SELECT CAST(oh.Pakbon_dat AS DATE) AS ds,
               SUM(osg.quantity)           AS y   -- TODO: adjust column
        FROM   Orhsrg osg
        JOIN   Orhkrg oh ON oh.Pakbon_Nr = osg.Pakbon_Nr
        GROUP  BY CAST(oh.Pakbon_dat AS DATE)
        """
        unit = "units"
    else:  # import_volume
        sql = """
        SELECT CAST(gm.Boek_dat AS DATE)   AS ds,
               SUM(gm.Aantal)              AS y
        FROM   Gbkmut gm
        WHERE  gm.Bud_vers = 'MRP'
          AND  gm.TransType = 'T'
          AND  gm.TransSubType = 'A'
        GROUP  BY CAST(gm.Boek_dat AS DATE)
        """
        unit = "units"
    return sql, unit


def parquet_file(metric: str) -> Path:
    return PARQUET_DIR / f"{metric}.parquet"


def parquet_fresh(p: Path) -> bool:
    if not p.exists():
        return False
    age_h = (dt.datetime.now() - dt.datetime.fromtimestamp(p.stat().st_mtime)).total_seconds() / 3600
    return age_h < PARQUET_TTL_H


def refresh_parquet(metric: str):
    sql, _ = sql_for(metric)
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    if df.empty:
        raise ValueError("No data from SQL Server.")
    df.to_parquet(parquet_file(metric), index=False)


def get_series(metric: str, freq: str) -> pd.DataFrame:
    """Return aggregated series via DuckDB."""
    pfile = parquet_file(metric)
    if not parquet_fresh(pfile):
        refresh_parquet(metric)

    con = duckdb.connect(database=":memory:", read_only=False)
    rule = {
        "M": "date_trunc('month', ds)",
        "Q": "date_trunc('quarter', ds)",
        "Y": "date_trunc('year', ds)",
    }[freq]

    query = f"""
    SELECT {rule} AS ds,
           SUM(y) AS y
    FROM   parquet_scan('{pfile.as_posix()}')
    GROUP  BY 1
    ORDER  BY 1
    """
    df = con.execute(query).fetch_df()
    df["ds"] = pd.to_datetime(df["ds"])
    return df


# ───────────────────────────────────────────────────────────────────────
# MODEL CACHING HELPERS
# ───────────────────────────────────────────────────────────────────────

def model_file(metric: str, freq: str, algo: str) -> Path:
    return MODEL_DIR / f"{metric}_{freq}_{algo}.pkl"


def df_sig(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


def load_model_if_valid(path: Path, sig: str):
    if not path.exists():
        return None
    obj = pickle.load(open(path, "rb"))
    return obj["model"] if obj.get("sig") == sig else None


def save_model(path: Path, model, sig: str):
    with open(path, "wb") as f:
        pickle.dump({"model": model, "sig": sig}, f)


# ───────────────────────────────────────────────────────────────────────
# FORECAST FUNCTIONS
# ───────────────────────────────────────────────────────────────────────

def prophet_fc(df: pd.DataFrame, periods: int, freq: str, m_cached=None):
    m = m_cached or Prophet(interval_width=0.8)
    if not m_cached:
        m.fit(df.rename(columns={"ds": "ds", "y": "y"}))
    future = m.make_future_dataframe(periods=periods, freq=freq)
    out = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)
    return out, m


def arima_fc(df: pd.DataFrame, periods: int, freq: str, m_cached=None):
    s = df.set_index("ds")["y"]
    model = m_cached or pm.auto_arima(s, seasonal=False, stepwise=True, suppress_warnings=True)
    y_pred, conf = model.predict(periods, return_conf_int=True, alpha=0.2)
    idx = pd.date_range(s.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=periods, freq=freq)
    out = pd.DataFrame({"ds": idx, "yhat": y_pred, "yhat_lower": conf[:, 0], "yhat_upper": conf[:, 1]})
    return out, model


# ───────────────────────────────────────────────────────────────────────
# LLM NARRATIVE
# ───────────────────────────────────────────────────────────────────────

def build_narrative(metric: str, freq: str, horizon: int, fc: pd.DataFrame) -> str:
    llm = Ollama(model=LLM_NAME, base_url=OLLAMA_URL)
    p = PromptTemplate.from_template(
        "Tóm tắt dự báo {metric} {horizon} kỳ ({freq}).\n{tbl}\n<120 từ tiếng Việt."
    )
    tbl = "\n".join(
        f"{r.ds.date()} {r.yhat:.0f} [{r.yhat_lower:.0f}-{r.yhat_upper:.0f}]"
        for r in fc.itertuples()
    )
    return llm.invoke(p.format(metric=metric, freq=freq, horizon=horizon, tbl=tbl)).strip()


# ───────────────────────────────────────────────────────────────────────
# API ENDPOINT
# ───────────────────────────────────────────────────────────────────────

@app.post("/forecast", response_model=Res)
def forecast(req: Req):
    try:
        df = get_series(req.metric, req.frequency)
        if df.empty:
            raise ValueError("Series empty.")
        sig = df_sig(df)
        algo = "prophet" if (req.model == "auto" and len(df) >= 36) else req.model
        if algo == "auto":
            algo = "arima"

        mp = model_file(req.metric, req.frequency, algo)
        cached_model = load_model_if_valid(mp, sig)

        if algo == "prophet":
            fc, model = prophet_fc(df, req.horizon, req.frequency, cached_model)
        else:
            fc, model = arima_fc(df, req.horizon, req.frequency, cached_model)

        if cached_model is None:
            save_model(mp, model, sig)

        preds = [
            Pred(
                period=fc.ds.iloc[i].strftime("%Y-%m") if req.frequency == "M"
                else f"{fc.ds.iloc[i].year}-Q{((fc.ds.iloc[i].month-1)//3)+1}"
                if req.frequency == "Q"
                else str(fc.ds.iloc[i].year),
                forecast=float(fc.yhat.iloc[i]),
                lower=float(fc.yhat_lower.iloc[i]),
                upper=float(fc.yhat_upper.iloc[i]),
            )
            for i in range(len(fc))
        ]

        narrative = None
        if req.use_llm:
            try:
                narrative = build_narrative(req.metric, req.frequency, req.horizon, fc)
            except Exception as e:
                narrative = f"LLM error: {e}"

        unit = "USD" if req.metric == "revenue" else "units"
        return Res(
            metric=req.metric,
            frequency=req.frequency,
            horizon=req.horizon,
            unit=unit,
            model_used=algo,
            predictions=preds,
            narrative=narrative,
        )
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))


# ───────────────────────────────────────────────────────────────────────
# LOCAL DEV RUNNER
# ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main_duck:app", host="0.0.0.0", port=8000, reload=True)
