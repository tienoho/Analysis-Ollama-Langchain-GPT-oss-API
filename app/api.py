from fastapi import FastAPI, HTTPException
from .schemas import Req, Res, Pred
from .data import get_series
from .forecasting import (
    model_file,
    df_sig,
    load_model_if_valid,
    save_model,
    prophet_fc,
    arima_fc,
)
from .narrative import build_narrative

app = FastAPI(title="DuckDB-Parquet Forecast API")


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
