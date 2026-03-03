from pydantic import BaseModel
from datetime import datetime

class ForecastLiveResponse(BaseModel):
    location: str
    observation_time_utc: datetime
    forecast_time_utc: datetime
    temp_now_c: float
    temp_pred_t_plus_24_c: float
    delta_c: float
    note: str