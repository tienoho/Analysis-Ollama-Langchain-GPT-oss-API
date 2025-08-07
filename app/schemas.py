from typing import Literal, List
from pydantic import BaseModel, conint

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
