"""
사용자 등록 및 임베딩 벡터 생성을 담당하는 컨트롤러
라우터에서 받은 요청을 처리하고 서비스 레이어와 연결
"""

import logging
from typing import Dict

from core.vector_database import list_similarities, list_users, reset_collections
from fastapi import HTTPException
from schemas.user_schema import BaseResponse, EmbeddingRegister
from services.users_post_service import register_user

logger = logging.getLogger(__name__)


async def db_user_list():
    result = await list_users()
    return {
        "code": "REGISTERD_ID_CHECKED",
        "data": {
            "ids": result.get("ids", []),
            "count": len(result.get("ids", [])),
            # "metadatas": result.get("metadatas", []),
        },
    }


async def db_similarity_list():
    result = await list_similarities()
    return {
        "code": "REGISTERD_SIMILARITY_CHECKED",
        "data": {
            "ids": result.get("ids", []),
            "count": len(result.get("ids", [])),
            "metadatas": (result.get("metadatas", [])),
        },
    }


async def db_reset_data():
    reset_collections()  # 동기 함수이므로 await 필요 없음
    return BaseResponse(status="success", code="CHROMADB_RESET_SUCCESS")


async def create_user(user_data: EmbeddingRegister) -> Dict:
    """
    새 사용자를 등록하고 임베딩 벡터를 생성하는 컨트롤러 함수

    Args:
        user_data: 사용자 등록 데이터 (Pydantic 모델)

    Returns:
        Dictionary containing the response code and result

    Raises:
        HTTPException: 오류 발생 시 적절한 상태 코드와 메시지를 포함한 예외 발생
    """
    try:
        result = await register_user(user_data)
        return {"code": "EMBEDDING_REGISTER_SUCCESS", "data": result}
    except HTTPException as http_ex:
        logger.warning(f"[REGISTER_USER_HTTP_ERROR] {http_ex.detail}")
        raise
    except Exception as e:
        logger.exception(
            f"[REGISTER_USER_FATAL_ERROR]: {str(e)}"
        )  # 자동 traceback 포함
        raise HTTPException(
            status_code=500,
            detail={
                "code": "EMBEDDING_REGISTER_SERVER_ERROR",
                "data": None,
            },
        )
