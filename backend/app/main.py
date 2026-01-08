from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.forecast import router as forecast_router

app = FastAPI(title="Air Pollutan Backend")

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*",]
)

# Routers
app.include_router(forecast_router)

@app.get("/")
def root():
    return {"status": "running"}