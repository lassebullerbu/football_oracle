import os
import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from engine import extract_club_features, predict_match_result_dict

app = FastAPI(title="Football Oracle Prediction API")

# --- 1. Load Assets use Path Relative for Docker Compatibility ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, "models", "football_stack_reg_model.pkl")
PIPELINE_PATH = os.path.join(BASE_PATH, "models", "football_pipeline.pkl")
CLUBS_PATH = os.path.join(BASE_PATH, "raw_data", "clubs.csv")
DATA_PATH = os.path.join(BASE_PATH, "raw_data", "processed_data.csv")

try:
    # Load Model and Pipeline
    model = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)
    # Load Clubs and Processed Data for API Logic
    clubs_df = pd.read_csv(CLUBS_PATH)
    processed_data = pd.read_csv(DATA_PATH)
    # Create Dictionary of Club Stats for API
    stats_dict = extract_club_features(processed_data, clubs_df)
    print("✅ API Assets & Stats Dictionary Loaded Successfully")
except Exception as e:
    print(f"❌ Error during asset loading: {str(e)}")

# --- 2. Request Schema ---
class PredictRequest(BaseModel):
    home_team: str
    away_team: str
    date: str

@app.get("/")
def root():
    return {"message": "Football Oracle API is online"}

@app.post("/predict")
async def predict(req: PredictRequest):
    # call Engine Logic to get prediction
    result = predict_match_result_dict(
        req.home_team, req.away_team, req.date,
        clubs_df, stats_dict, pipeline, model
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result

if __name__ == "__main__":
    # Cloud Run port 8080 is the standard value
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
