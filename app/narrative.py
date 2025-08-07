from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import pandas as pd

from .config import LLM_NAME, OLLAMA_URL


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
