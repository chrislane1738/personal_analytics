"""FastAPI application entry point."""
from fastapi import FastAPI
from api.routes import predictions, fighters, events, model_stats

app = FastAPI(title="UFC Fight Prediction API", version="1.0.0")
app.include_router(predictions.router, prefix="/api/predictions", tags=["Predictions"])
app.include_router(fighters.router, prefix="/api/fighters", tags=["Fighters"])
app.include_router(events.router, prefix="/api/events", tags=["Events"])
app.include_router(model_stats.router, prefix="/api/model", tags=["Model"])

@app.get("/health")
def health_check():
    return {"status": "ok", "model_a_loaded": False, "model_b_loaded": False}
