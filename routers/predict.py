import os

import joblib
import numpy as np
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# 定数の定義
MODEL_FILE_PATH = 'titanic_model.pkl'

router = APIRouter()
templates = Jinja2Templates(directory="templates")

def load_model():
    if not os.path.exists(MODEL_FILE_PATH):
        raise FileNotFoundError(f"モデルファイル {MODEL_FILE_PATH} が見つかりませんでした.")
    return joblib.load(MODEL_FILE_PATH)

@router.post("/predict", response_class=HTMLResponse)
async def predict_survival(request: Request,
                           Pclass: int = Form(...),
                           Sex: int = Form(...),
                           Age: float = Form(...),
                           SibSp: int = Form(...),
                           Parch: int = Form(...),
                           Fare: float = Form(...),
                           Embarked: int = Form(...)):
    try:
        model = load_model()

        features = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])
        prediction = model.predict(features)
        survival = "Survived" if prediction[0] == 1 else "Did not survive"
        return templates.TemplateResponse("form.html", {"request": request, "result": f"推論結果: {survival}"})
    except Exception as e:
        return templates.TemplateResponse("form.html", {"request": request, "result": f"推論に失敗しました。: {str(e)}"})
