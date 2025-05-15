# 임베딩 관련 데이터 모델

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class UserEmbedding(BaseModel):
    """
    사용자 임베딩 벡터 데이터 모델
    """

    userId: int = Field(..., description="사용자 ID")
    emailDomain: str = Field(..., description="이메일 도메인")
    embedding: List[float] = Field(..., description="임베딩 벡터")
    metadata: Dict[str, Any] = Field(..., description="사용자 메타데이터")

    class Config:
        schema_extra = {
            "example": {
                "userId": 1,
                "emailDomain": "kakaotech.com",
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],  # 실제로는 더 큰 차원의 벡터
                "metadata": {
                    "gender": "남자",
                    "ageGroup": "AGE_20S",
                    "MBTI": "ESTP",
                    "interests": ["BAKING", "DRAWING", "GAMING"],
                },
            }
        }


class UserSimilarities(BaseModel):
    """
    사용자 간 유사도 정보 모델
    """

    userId: int = Field(..., description="기준 사용자 ID")
    similarities: Dict[str, float] = Field(
        ..., description="다른 사용자와의 유사도 점수"
    )

    class Config:
        schema_extra = {
            "example": {"userId": 1, "similarities": {"2": 0.92, "3": 0.85, "4": 0.76}}
        }


class CategoryWeights(BaseModel):
    """
    카테고리별 유사도 계산 가중치 모델
    """

    category: str = Field(..., description="카테고리 이름")
    weights: Dict[str, float] = Field(..., description="각 속성별 가중치")

    class Config:
        schema_extra = {
            "example": {
                "category": "FRIEND",
                "weights": {
                    "personality": 1.5,
                    "preferredPeople": 1.2,
                    "hobbies": 1.0,
                    "currentInterests": 0.8,
                    "MBTI": 0.7,
                },
            }
        }


class EmbeddingResponse(BaseModel):
    """
    임베딩 관련 API 응답 모델
    """

    code: str = Field(..., description="응답 코드")
    data: Optional[Union[Dict, List, UserEmbedding, UserSimilarities]] = Field(
        None, description="응답 데이터"
    )

    class Config:
        schema_extra = {
            "example": {
                "code": "EMBEDDING_OPERATION_SUCCESS",
                "data": {"userId": 1, "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]},
            }
        }
