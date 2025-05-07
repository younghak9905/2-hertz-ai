"""
매칭 추천 기능을 담당하는 컨트롤러
사용자 간 유사도 기반 매칭을 처리하고 결과를 반환
"""

import logging
from typing import Dict

from fastapi import HTTPException

from app.services.tuning_service import get_matching_users

logger = logging.getLogger(__name__)


async def get_tuning_matches(user_id: str) -> Dict:
    """
    사용자 ID를 기반으로 매칭 추천을 제공하는 컨트롤러 함수

    Args:
        userId: 매칭을 요청한 사용자의 ID

    Returns:
        Dictionary containing the response code and matching user IDs list

    Raises:
        HTTPException: 오류 발생 시 적절한 상태 코드와 메시지를 포함한 예외 발생
    """
    try:
        result = await get_matching_users(user_id)

        if not result:
            return {"code": "TUNING_SUCCESS_BUT_NO_MATCH", "data": None}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in tuning controller: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"code": "TUNING_INTERNAL_SERVER_ERROR", "data": None},
        )

    return {
        "code": "TUNING_SUCCESS",
        "data": {
            "userIdList": result,
        },
    }
