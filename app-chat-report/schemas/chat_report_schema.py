"""
신고 기능 관련 데이터 모델 정의
유저의 메세지 신고 - 제재 여부 판단에 대한 Pydantic 모델
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ChatReportRequest(BaseModel):
    """
    신고된 메세지와 이유를 전송하여 제재 대상에 대한 판단을 요청하는 스키마 모델
    """

    # messageId: int = Field(..., description="메세지 아이디")
    messageContent: str = Field(..., description="신고된 메세지 내용")
    # reportedUserId: int = Field(..., description="신고된 유저")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "messageId": 5555,
                "messageContent": "욕설이 포함된 메시지 내용입니다.",
                "reportedUserId": 20,
            }
        }
    )


class ChatReportResponse(BaseModel):
    """
    신고된 내용에 대해 AI 모델이 자동 판별 후 반환하는 스키마 모델
    부적절한 내용이 맞을 경우 True 반환 / 부적절하지 않다면  False 반환
    """

    message: str = Field(..., description="신고된 채팅 메세지 내용")
    isToxic: bool = Field(..., description="유해성 여부 판단")  # True if inappropriate
    # confidence: float = Field(..., description="신뢰도")

    model_config = ConfigDict(
        json_schema_extra={
            "example": [{"code": "CENSORED_SUCCESS", "data": {"result": True}}]
        }
    )
