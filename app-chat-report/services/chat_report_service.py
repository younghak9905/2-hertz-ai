import json

from fastapi import HTTPException
from models.hyper_clover_loader import ModelSingleton
from models.kcelectra_base_loader import get_model
from schemas.chat_report_schema import ChatReportRequest, ChatReportResponse
from utils.logger import log_performance, logger


@log_performance(operation_name="handle_chat_report", include_memory=True)
async def handle_chat_report_(message: str) -> ChatReportResponse:
    try:
        clova_model = ModelSingleton.get_instance()
        result = clova_model.classify(text=message)
        isToxic = any(
            keyword in result
            for keyword in ["유해합니다", "제재 대상", "부적절", "비속어"]
        )

        return ChatReportResponse(message=message, isToxic=isToxic)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@log_performance(operation_name="handle_chat_report", include_memory=True)
async def handle_chat_reports(body: ChatReportRequest) -> ChatReportResponse:
    try:
        model = get_model()
        result = model(body.messageContent)[0]
        label = result.get("label", "")
        confidence_raw = result.get("score", "")
        if confidence_raw != "":
            confidence = round(float(confidence_raw), 4)
        else:
            confidence = 0.0  # 또는 None

        mornitoring_yn = "Y" if confidence < 0.75 else "N"  # 모니터링 필요성

        is_toxic = label != "LABEL_0"

        response = ChatReportResponse(
            code="CENSORED_SUCCESS",
            data={
                "result": is_toxic,
                "label": label,
                "confidence": confidence,
                "monitoring": mornitoring_yn,
            },
        )
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
