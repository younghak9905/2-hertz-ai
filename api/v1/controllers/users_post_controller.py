"""
사용자 등록 및 임베딩 벡터 생성을 담당하는 컨트롤러
라우터에서 받은 요청을 처리하고 서비스 레이어와 연결
"""

from typing import Dict

from fastapi import HTTPException

# TODO: 실제 서비스 연결
# from services.users_post_service import register_user
from schemas.user_schema import EmbeddingRegister


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
        # 모의 응답 구현 (실제 서비스 연결 시 교체될 예정)
        # 실제 구현에서는 아래 코드를 주석 해제하고 모의 응답 코드를 제거
        # TODO:return await register_user(user_data)

        # 사용자 ID 중복 체크 시뮬레이션 (실제로는 서비스 레이어에서 처리)
        if user_data.userId == 999:  # 이미 존재하는 ID로 가정
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "EMBEDDING_CONFLICT_DUPLICATE_ID",
                    "data": None,
                },
            )

        # 성공 케이스 시뮬레이션
        print(f"Creating user with data: {user_data.model_dump()}")
        return {"code": "EMBEDDING_REGISTER_SUCCESS", "data": None}
    except HTTPException:
        # HTTP 예외(400)는 그대로 전달
        raise
    except Exception as e:
        # TODO:로깅 추가 (실제 구현에서는 logger 사용)
        print(f"Error in user registration controller: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "EMBEDDING_REGISTER_SERVER_ERROR",
                "data": None,
            },
        )
