from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd
import requests
import joblib
from catboost import CatBoostClassifier
from io import StringIO

app = FastAPI()

# Load models
catboost_model = CatBoostClassifier()
catboost_model.load_model("catboost_model.cbm")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
narration_classifier = joblib.load("narration_classifier.pkl")

# Define input schema
class PredictRequest(BaseModel):
    file_url: str

@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        file_url = request.file_url

        # Fetch file from URL
        response = requests.get(file_url)
        if response.status_code != 200:
            return {"error": f"Failed to fetch file from URL: {file_url}"}

        # Read CSV with comma separator and thousands format
        df = pd.read_csv(
            StringIO(response.text),
            sep=",",
            encoding="utf-8",
            thousands=","
        )

        # Handle missing numeric fields
        df["Net Amount"] = df["Net Amount"].replace(",", "", regex=True).astype(float)
        narration_score = narration_classifier.predict_proba(
            tfidf_vectorizer.transform(df["Line Desc"].fillna(""))
        )[:, 1]
        df["narration_risk_score"] = narration_score

        # Categorical + numerical features
        cat_cols = ["Account Name", "Source Name", "Batch Name"]
        num_cols = ["Net Amount", "narration_risk_score"]
        for col in cat_cols:
            df[col] = df[col].fillna("MISSING").astype(str)

        features = cat_cols + num_cols

        # Predict
        predictions = catboost_model.predict(df[features])
        probabilities = catboost_model.predict_proba(df[features])[:, 1]

        # Return top 10 results
        df["Predicted Risk"] = predictions
        df["Predicted Probability"] = probabilities
        return df[["Predicted Risk", "Predicted Probability"]].head(10).to_dict(orient="records")

    except Exception as e:
        return {"error": str(e)}
