"""
매칭 추천 기능을 담당하는 컨트롤러
사용자 간 유사도 기반 매칭을 처리하고 결과를 반환
"""

from typing import Dict

from fastapi import HTTPException

# 실제 서비스는 나중에 구현하고, 지금은 모의 응답을 사용합니다
# from services.tuning_service import get_matching_users


async def get_tuning_matches(user_id: int) -> Dict:
    """
    사용자 ID를 기반으로 매칭 추천을 제공하는 컨트롤러 함수

    Args:
        user_id: 매칭을 요청한 사용자의 ID

    Returns:
        Dictionary containing the response code and matching user IDs list

    Raises:
        HTTPException: 오류 발생 시 적절한 상태 코드와 메시지를 포함한 예외 발생
    """
    try:
        # 모의 응답 구현 (실제 서비스 연결 시 교체될 예정)
        # 실제 구현에서는 아래 코드를 주석 해제하고 모의 응답 코드를 제거
        # TODO:return await get_matching_users(user_id)

        # 테스트용 샘플 데이터 - ID가 짝수인 경우 매칭 결과가 있음
        if user_id % 2 == 0:
            return {
                "code": "TUNING_SUCCESS",
                "data": {"userIdList": [30, 1, 5, 6, 99, 56]},
            }
        # ID가 홀수인 경우 매칭 결과가 없음 (매칭 가능한 사용자 없음)
        else:
            return {"code": "TUNING_SUCCESS_BUT_NO_MATCH", "data": None}
    except HTTPException:
        # HTTP 예외는 그대로 전달
        raise
    except Exception as e:
        # TODO:로깅 추가 (실제 구현에서는 logger 사용)
        print(f"Error in tuning controller: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "TUNING_INTERNAL_SERVER_ERROR",
                "data": None,
            },
        )
