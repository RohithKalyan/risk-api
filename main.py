from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import requests
import joblib
from io import StringIO

app = FastAPI()

# Load TF-IDF vectorizer and narration classifier
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
narration_classifier = joblib.load("narration_classifier.pkl")

# Define input schema
class PredictionRequest(BaseModel):
    file_url: str

@app.post("/predict")
async def predict(req: PredictionRequest):
    try:
        file_url = req.file_url.strip()

        # 🔍 Print debug info to Render logs
        print(">>>> Received request at /predict")
        print(">>>> File URL:", file_url)

        response = requests.get(file_url)
        response.raise_for_status()

        # Read CSV from URL
        df = pd.read_csv(StringIO(response.text))

        # TF-IDF prediction
        narration_scores = narration_classifier.predict_proba(
            tfidf_vectorizer.transform(df["Line Desc"].fillna(""))
        )[:, 1]

        df["Narration Risk Score"] = narration_scores

        return df[["Line Desc", "Narration Risk Score"]].head(10).to_dict(orient="records")

    except Exception as e:
        print(">>>> ERROR:", str(e))  # Optional: log error to logs as well
        return {"error": str(e)}
