# 매칭 관련 라우터

from typing import Optional

from fastapi import APIRouter, Query

from schemas.tuning_schema import TuningResponse

from ..controllers import tuning_controller


class TuningRouter:
    def __init__(self):
        self.router = APIRouter(prefix="/api", tags=["tuning"])
        self.router.add_api_route("/v1/tuning", self.get_tuning, methods=["GET"])

    # @router.get("/api/v1/tuning/{user_id}", response_model=TuningResponse)
    async def get_tuning(
        self, user_id: int = Query(..., description="매칭할 사용자의 ID", gt=0)
    ) -> TuningResponse:
        """
        사용자 ID를 기반으로 매칭 추천을 제공합니다.

        - **user_id**: 매칭을 요청한 사용자의 ID (1 이상의 정수)

        **응답 예시**:
        ```json
        {
          "code": "TUNING_SUCCESS",
          "data": {
            "userIdList": [30, 1, 5, 6, 99, 56]
          }
        }
        ```

        매칭 결과가 없는 경우:
        ```json
        {
          "code": "TUNING_SUCCESS_BUT_NO_MATCH",
          "data": null
        }
        ```
        """
        return await tuning_controller.get_tuning_matches(user_id)
