import hashlib
import pickle
import pandas as pd
import pmdarima as pm
from prophet import Prophet
from pathlib import Path

from .config import MODEL_DIR


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
