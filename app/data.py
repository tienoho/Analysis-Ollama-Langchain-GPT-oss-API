import datetime as dt
import pandas as pd
import duckdb
from pathlib import Path
from sqlalchemy import text

from .config import PARQUET_DIR, PARQUET_TTL_H, engine


def sql_for(metric: str) -> tuple[str, str]:
    if metric == "revenue":
        sql = """
        SELECT CAST(okr.OrderDat AS DATE) AS ds,
               SUM(osg.Extra_pr)          AS y
        FROM   Orsrg osg
        JOIN   Orkrg okr ON okr.OrderNr = osg.OrderNr
        WHERE  okr.Or_Soort = 'V'
        GROUP  BY CAST(okr.OrderDat AS DATE)
        """
        unit = "USD"
    elif metric == "production_output":
        sql = """
        SELECT CAST(oh.Pakbon_dat AS DATE) AS ds,
               SUM(osg.Aantal)            AS y
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
