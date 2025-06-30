import json

from fastapi import HTTPException
from models.hyper_clover_loader import ModelSingleton
from models.kcelectra_base_loader import get_model
from schemas.chat_report_schema import ChatReportResponse
from utils import logger


@logger.log_performance(operation_name="handle_chat_report", include_memory=True)
async def handle_chat_report_(message: str) -> ChatReportResponse:
    try:
        clova_model = ModelSingleton.get_instance()
        result = clova_model.classify(text=message)
        isToxic = any(
            keyword in result
            for keyword in ["유해합니다", "제재 대상", "부적절", "비속어"]
        )

        return ChatReportResponse(
            message=message,
            isToxic=isToxic,
            # confidence=result.get(
            #     "confidence", 0.9
            # ),  # confidence 키가 없으면 기본값 사용
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@logger.log_performance(operation_name="handle_chat_report", include_memory=True)
async def handle_chat_reports(message: str) -> ChatReportResponse:
    try:
        model = get_model()
        result = model(message)[0]
        print(result)
        label = result.get("label", "")

        is_toxic = label in ["LABEL_1", "toxic", "유해", "HATE", "NSFW", "bad"]
        response = ChatReportResponse(message=message, isToxic=is_toxic)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
