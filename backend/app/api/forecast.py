from fastapi import APIRouter, Query
from app.services.forecast_service import get_forecast

router = APIRouter(
    prefix="/forecast",
    tags=["forecast"]
)

@router.get("/")
def forecast(
    model: str = Query(..., regex="^(tft|nbeats)")
):
    return get_forecast(model)