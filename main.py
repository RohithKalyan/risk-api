from fastapi import FastAPI, Request
import pandas as pd
import requests
import joblib
from catboost import CatBoostClassifier
from io import StringIO
from pydantic import BaseModel

app = FastAPI()

class PredictionRequest(BaseModel):
    file_url: str

# Load models
catboost_model = CatBoostClassifier()
catboost_model.load_model("catboost_model.cbm")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
narration_classifier = joblib.load("narration_classifier.pkl")

@app.post("/predict")
async def predict(req: PredictionRequest):
    try:
        file_url = req.file_url.strip()

        # Download CSV
        response = requests.get(file_url)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))

        # Clean Net Amount
        df["Net Amount"] = df["Net Amount"].replace(",", "", regex=True).astype(float)

        # Narration classifier score
        df["narration_risk_score"] = narration_classifier.predict_proba(
            tfidf_vectorizer.transform(df["Line Desc"].fillna(""))
        )[:, 1]

        # Define features
        cat_cols = ["Account Name", "Source Name", "Batch Name"]
        num_cols = ["Net Amount", "narration_risk_score"]

        # Preprocess categorical columns properly
        for col in cat_cols:
            df[col] = df[col].astype(str).fillna("MISSING").apply(lambda x: str(x).strip())

        # Construct feature DataFrame
        X = df[cat_cols + num_cols].copy()
        for col in cat_cols:
            X[col] = X[col].astype(str)  # enforce string type at prediction

        # Predict
        predictions = catboost_model.predict(X)
        probabilities = catboost_model.predict_proba(X)[:, 1]

        # Attach results
        df["Predicted Risk"] = predictions
        df["Predicted Probability"] = probabilities

        return df[["Predicted Risk", "Predicted Probability"]].head(10).to_dict(orient="records")

    except Exception as e:
        return {"error": str(e)}
