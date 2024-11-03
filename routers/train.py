import joblib
import pandas as pd
from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 定数の定義
MODEL_FILE_PATH = 'titanic_model.pkl'
RANDOM_SEED = 42

router = APIRouter()
templates = Jinja2Templates(directory="templates")

def preprocess_data(df):
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'].fillna('S', inplace=True)
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Fare'].fillna(df['Fare'].mean(), inplace=True)
    return df

@router.get("/train", response_class=HTMLResponse)
async def get_train_page(request: Request):
    return templates.TemplateResponse("train.html", {"request": request})

@router.post("/train", response_class=HTMLResponse)
async def train_model(request: Request, file: UploadFile):
    try:
        df = pd.read_csv(file.file)
        df = preprocess_data(df)

        X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
        y = df['Survived']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

        model = RandomForestClassifier(random_state=RANDOM_SEED)
        model.fit(X_train, y_train)

        # モデルの精度を計算
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # モデルの保存
        joblib.dump(model, MODEL_FILE_PATH)

        return templates.TemplateResponse("train.html", {
            "request": request,
            "message": "モデルの訓練に成功しました!",
            "accuracy": f"予測精度: {accuracy * 100:.2f}%"
        })
    except Exception as e:
        return templates.TemplateResponse("train.html", {"request": request, "message": f"訓練に失敗しました。: {str(e)}"})
