# 매칭 관련 컨트롤러 (사용자 간 유사도 기반 매칭 추천)

from typing import Dict, List, Optional

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
        # return await get_matching_users(user_id)

        # 테스트용 샘플 데이터 - 항상 매칭 결과가 있는 경우
        if user_id % 2 == 0:
            return {
                "code": "TUNING_SUCCESS",
                "data": {"userIdList": [30, 1, 5, 6, 99, 56]},
            }
        # 매칭 결과가 없는 경우
        else:
            return {"code": "TUNING_SUCCESS_BUT_NO_MATCH", "data": None}
    except Exception as e:
        # 로깅 추가 (실제 구현에서는 logger 사용)
        print(f"Error in tuning controller: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


async def get_tuning_matches_by_category(user_id: int, category: str) -> Dict:
    """
    사용자 ID와 카테고리를 기반으로 매칭 추천을 제공하는 컨트롤러 함수

    Args:
        user_id: 매칭을 요청한 사용자의 ID
        category: 매칭 카테고리 (예: FRIEND, MENTOR 등)

    Returns:
        Dictionary containing the response code and matching user IDs list

    Raises:
        HTTPException: 오류 발생 시 적절한 상태 코드와 메시지를 포함한 예외 발생
    """
    try:
        # 모의 응답 구현 (실제 서비스 연결 시 교체될 예정)
        # 실제 구현에서는 아래 코드를 주석 해제하고 모의 응답 코드를 제거
        # return await get_matching_users_by_category(user_id, category)

        # 카테고리 유효성 검사 (예시)
        valid_categories = ["FRIEND", "MENTOR", "ACTIVITY", "STUDY"]
        if category not in valid_categories:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid category: {category}. Must be one of {valid_categories}",
            )

        # 테스트용 샘플 데이터 - 항상 매칭 결과가 있는 경우 (카테고리에 따라 다른 결과)
        if category == "FRIEND":
            return {"code": "TUNING_SUCCESS", "data": {"userIdList": [30, 1, 5]}}
        elif category == "MENTOR":
            return {"code": "TUNING_SUCCESS", "data": {"userIdList": [42, 87, 21]}}
        # 다른 카테고리는 매칭 결과 없음
        else:
            return {"code": "TUNING_SUCCESS_BUT_NO_MATCH", "data": None}
    except HTTPException:
        # HTTP 예외는 그대로 전달
        raise
    except Exception as e:
        # 로깅 추가 (실제 구현에서는 logger 사용)
        print(f"Error in tuning controller: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
