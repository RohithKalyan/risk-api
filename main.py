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

        # Download CSV
        response = requests.get(file_url)
        response.raise_for_status()
        print(">>> File download successful")

        try:
            df = pd.read_csv(StringIO(response.text), encoding='utf-8')
        except Exception as e:
            print(">>> CSV Read Error:", str(e))
            return JSONResponse(content={"error": "Failed to read CSV file. Ensure it's UTF-8 encoded."}, status_code=400)

        print(f">>> Original rows: {len(df)}")

        if "Line Desc" not in df.columns:
            return JSONResponse(content={"error": "Missing 'Line Desc' column in uploaded CSV."}, status_code=400)

        df = df.head(3)
        print(f">>> Using rows: {len(df)}")

        # Fill NA and check for empty content
        narrations = df["Line Desc"].fillna("")
        if narrations.str.strip().eq("").all():
            return JSONResponse(content={"error": "All values in 'Line Desc' are empty."}, status_code=400)

        tfidf_features = tfidf_vectorizer.transform(narrations)
        scores = narration_classifier.predict_proba(tfidf_features)[:, 1]

        df["Narration Risk Score"] = scores
        result_data = df[["Line Desc", "Narration Risk Score"]].to_dict(orient="records")

        print(f">>> Returning {len(result_data)} rows")
        return JSONResponse(content={"results": result_data}, media_type="application/json")

    except Exception as e:
        print(f">>> Unhandled error: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
