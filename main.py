from fastapi import FastAPI, Request
import pandas as pd
import requests
import joblib
from catboost import CatBoostClassifier
from io import StringIO

app = FastAPI()

catboost_model = CatBoostClassifier()
catboost_model.load_model("catboost_model.cbm")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
narration_classifier = joblib.load("narration_classifier.pkl")

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    file_url = data.get("file_url")
    if not file_url:
        return {"error": "Missing file_url"}

    response = requests.get(file_url)
    df = pd.read_csv(StringIO(response.text))

    df["Net Amount"] = df["Net Amount"].replace(",", "", regex=True).astype(float)
    narration_score = narration_classifier.predict_proba(
        tfidf_vectorizer.transform(df["Line Desc"].fillna(""))
    )[:, 1]
    df["narration_risk_score"] = narration_score

    cat_cols = ["Account Name", "Source Name", "Batch Name"]
    num_cols = ["Net Amount", "narration_risk_score"]
    for col in cat_cols:
        df[col] = df[col].fillna("MISSING").astype(str)

    features = cat_cols + num_cols
    predictions = catboost_model.predict(df[features])
    probabilities = catboost_model.predict_proba(df[features])[:, 1]

    df["Predicted Risk"] = predictions
    df["Predicted Probability"] = probabilities

    return df[["Predicted Risk", "Predicted Probability"]].head(10).to_dict(orient="records")
