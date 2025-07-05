from datetime import datetime, timedelta, timezone

from db.mongodb import mongodb
from fastapi import HTTPException
from models.hyper_clover_loader import ModelSingleton
from models.kcelectra_base_loader import get_model
from schemas.chat_report_schema import ChatReportRequest, ChatReportResponse
from utils.logger import log_performance, logger

# mongodb 인스턴스에서 컬렉션 가져오기
chat_report_collection = mongodb.get_collection()


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

        monitoring_yn = "Y" if confidence < 0.75 else "N"  # 모니터링 필요성

        is_toxic = label != "LABEL_0"

        response = ChatReportResponse(
            code="CENSORED_SUCCESS",
            data={
                "result": is_toxic,
                "label": label,
                "confidence": confidence,
                "monitoring": monitoring_yn,
            },
        )

        # MongoDB에 저장할 문서(Document) 생성
        report_data = {
            "messageId": body.messageId,
            "messageContent": body.messageContent,
            "result": is_toxic,
            "label": label,
            "confidence": confidence,
            "monitoring": monitoring_yn,
            "report_time": (datetime.utcnow() + timedelta(hours=9))
            .replace(microsecond=0)
            .isoformat(),  # UTC + 9시간 (Seoul 시간)
        }

        # MongoDB에 문서 삽입
        chat_report_collection.insert_one(report_data)

        logger.info(
            f"Chat report processed: {body.messageId}, Result: {is_toxic}, Label: {label}, Confidence: {confidence}"
        )
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
