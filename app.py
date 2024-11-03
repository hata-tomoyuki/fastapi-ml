from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from routers import predict, train

# FastAPIインスタンスの作成
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 静的ファイルのマウント
app.mount("/static", StaticFiles(directory="static"), name="static")

# ルーターの登録
app.include_router(train.router)
app.include_router(predict.router)

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})
