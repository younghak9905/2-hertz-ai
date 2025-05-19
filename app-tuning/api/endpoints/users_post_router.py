"""
사용자 등록 API 엔드포인트 정의 및 관리
사용자 개인정보, 관심사, 키워드를 받아 임베딩 벡터를 생성하고 벡터 DB에 저장
/api 요청을 처리하고, 비즈니스 로직 실행을 위해 컨트롤러와 연결
"""

from fastapi import APIRouter, Body
from schemas.user_schema import BaseResponse, EmbeddingRegister

from ..controllers import users_post_controller


class UserPostRouter:
    """
    사용자 등록 및 사용자 관련 엔드포인트를 처리하는 라우터 클래스
    사용자 관련 API 경로 정의
    """

    def __init__(self):
        # 라우터 생성
        self.router = APIRouter(prefix="/api", tags=["users"])
        # 엔드포인트 등록 (/api/v1/users)
        self.router.add_api_route(
            "/v1/users",
            self.create_user,
            methods=["POST"],
            response_model=BaseResponse,
            summary="사용자 등록",
            description="사용자 등록 후 임베딩 벡터를 생성합니다.",
        )

        self.router.add_api_route(
            "/v1/users",
            self.db_user_list,
            methods=["GET"],
            response_model=BaseResponse,
            summary="사용자 조회(내부 확인용)",
            description="벡터DB user_collection에 등록된 사용자 리스트를 조회합니다.",
        )
        self.router.add_api_route(
            "/v1/similarities",
            self.db_similarity_list,
            methods=["GET"],
            response_model=BaseResponse,
            summary="매칭 스코어 조회(내부 확인용)",
            description="벡터DB-similarity_collection에 등록된 모든 매칭 스코어를 조회합니다.",
        )
        self.router.add_api_route(
            "/v1/reset/chromadb",
            self.db_reset_data,
            methods=["DELETE"],
            response_model=BaseResponse,
            summary="(테스트 환경 리셋을 위한 용도)",
            description="벡터DB-collection에 저장된 모든 데이터를 초기화 합니다.",
        )

    async def db_user_list(self) -> BaseResponse:
        return await users_post_controller.db_user_list()

    async def db_similarity_list(self) -> BaseResponse:
        return await users_post_controller.db_similarity_list()

    async def db_reset_data(self) -> BaseResponse:
        return await users_post_controller.db_reset_data()

    async def create_user(
        self, user_data: EmbeddingRegister = Body(..., description="사용자 등록 데이터")
    ) -> BaseResponse:
        """
        신규 사용자를 등록 후 임베딩 벡터 생성

        - **user_data**: 사용자 등록 정보 (개인정보, 키워드, 관심사 등)

        **응답 예시**:
        ```json
        {
          "code": "EMBEDDING_REGISTER_SUCCESS",
          "data": null
        }
        ```

        **오류 응답**:
        - 409 Conflict: 이미 존재하는 사용자 ID
        ```json
        {
          "code": "EMBEDDING_REGISTER_CONFLICT",
          "data": null
        }
        ```
        """
        return await users_post_controller.create_user(user_data)
