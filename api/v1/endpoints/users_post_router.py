# 사용자 등록 라우터

from typing import Dict

from fastapi import APIRouter, Body

from schemas.user_schema import BaseResponse, EmbeddingRegister

from ..controllers import users_post_controller


class UserPostRouter:
    def __init__(self):
        self.router = APIRouter(prefix="/api", tags=["users"])
        self.router.add_api_route("/v1/users", self.create_user, methods=["POST"])

    # @router.post("", response_model=BaseResponse, status_code=201)
    async def create_user(
        self, user_data: EmbeddingRegister = Body(..., description="사용자 등록 데이터")
    ):
        """
        새 사용자를 등록하고 임베딩 벡터를 생성합니다.

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
          "message": "User with ID 999 already exists"
        }
        ```
        """
        return await users_post_controller.create_user(user_data)
