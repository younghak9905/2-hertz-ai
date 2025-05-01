# -*- coding: utf-8 -*-

# 애플리케이션 초기화 및 구성 / 서버 시작점 역할 수행

import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from api.v1.endpoints.tuning_router import TuningRouter
from api.v1.endpoints.users_post_router import UserPostRouter

load_dotenv()

# FastAPI 앱 인스턴스 생성
app = FastAPI(
    title="TUNING API",
    description="조직 내부 사용자 간의 자연스럽고 부담 없는 소통을 돕는 소셜 매칭 서비스 API",
    version="1.0.0",
)


# 라우터 등록
app.include_router(UserPostRouter().router)
app.include_router(TuningRouter().router)


# 개발 환경에서 Root Path 접속 시, swagger 페이지로 Redirect
@app.get("/")
async def root():
    env = os.getenv("ENVIRONMENT")
    if env == "dev":
        return RedirectResponse(url="/docs")
    else:
        return HTTPException(status_code=404)
