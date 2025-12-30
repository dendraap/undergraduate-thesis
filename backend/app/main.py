from fastapi import FastAPI
from app.api.forecast import router as forecast_router

app = FastAPI(title="Air Pollutan Backend")
app.include_router(forecast_router)

@app.get("/")
def root():
    return {"status": "running"}