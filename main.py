from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import requests
import joblib
from io import StringIO

app = FastAPI()

# Load models
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
narration_classifier = joblib.load("narration_classifier.pkl")

# Input schema
class PredictionRequest(BaseModel):
    file_url: str

@app.post("/predict")
async def predict(req: PredictionRequest):
    try:
        print(">>> Received request at /predict")

        file_url = req.file_url.strip()
        print(f">>> File URL: {file_url}")

        # Download and parse
        response = requests.get(file_url)
        response.raise_for_status()
        print(">>> File download successful")

        df = pd.read_csv(StringIO(response.text))
        print(f">>> Original rows: {len(df)}")
        
        df = df.head(3)  # Limit to 3 for testing
        print(f">>> Using rows: {len(df)}")

        # Transform and predict
        narrations = df["Line Desc"].fillna("")
        tfidf_features = tfidf_vectorizer.transform(narrations)
        scores = narration_classifier.predict_proba(tfidf_features)[:, 1]

        df["Narration Risk Score"] = scores
        result = df[["Line Desc", "Narration Risk Score"]].to_dict(orient="records")
        
        print(f">>> Returning {len(result)} rows")
        return JSONResponse(content=result, media_type="application/json")

    except Exception as e:
        print(f">>> Error: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
