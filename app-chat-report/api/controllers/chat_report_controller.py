# 튜닝리포트(뉴스) 생성 컨트롤러
from fastapi import HTTPException
from schemas.chat_report_schema import ChatReportRequest, ChatReportResponse
from services.chat_report_service import handle_chat_reports
from utils.logger import logger


async def handle_chat_report(messageContent: str) -> ChatReportResponse:
    result = await handle_chat_reports(messageContent)

    if isinstance(result, dict):
        return ChatReportResponse(**result)
    elif isinstance(result, ChatReportResponse):
        return result
    else:
        raise HTTPException(status_code=500, detail=f"Invalid response type: {type(result)}")