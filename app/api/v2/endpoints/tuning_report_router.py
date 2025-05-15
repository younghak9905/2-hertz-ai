# 튜닝리포트(뉴스) 생성 관련 엔드포인트
from fastapi import APIRouter, Request

from app.api.v2.controllers.tuning_report_controller import TuningReportController
from app.schemas.tuning_schema import TuningReport, TuningReportResponse

# APIRouter 인스턴스 생성
router = APIRouter()


class TuningReportRouter:
    """
    사용자 매칭 뉴스 생성 엔드포인트를 처리하는 라우터 클래스
    뉴스 관련 API 경로 정의
    """

    def __init__(self):
        # 라우터 생성
        self.router = APIRouter(prefix="/api", tags=["report"])
        self.router.add_api_route(
            "/v2/report",
            self.create_tuning_report,
            methods=["POST"],
            response_model=TuningReportResponse,
            summary="튜닝 리포트(뉴스) 생성 요청",
            description="해당 사용자 정보를 기반으로 매칭 리포트를 생성합니다.",
        )

    async def create_tuning_report(self, request: Request, body: TuningReport):
        controller = TuningReportController(app=request.app)
        return await controller.create_tuning_report(body)
