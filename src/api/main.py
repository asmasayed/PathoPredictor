"""
API main file for PathoPredictor service.
"""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="PathoPredictor API")

class PredictionRequest(BaseModel):
    """Request model for predictions."""
    sequence: str
    host_metadata: dict
    time_series: list

@app.get("/")
def root():
    """Root endpoint."""
    return {"message": "PathoPredictor API"}

@app.post("/predict")
def predict(request: PredictionRequest):
    """
    Main prediction endpoint.
    
    Args:
        request: Prediction request with sequence, metadata, and time series
        
    Returns:
        Prediction results
    """
    # Prediction logic here
    return {"prediction": "placeholder"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
