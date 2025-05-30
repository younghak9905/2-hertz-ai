# 튜닝리포트(뉴스) 생성 컨트롤러

from fastapi import HTTPException

from ...schemas.tuning_schema import TuningReport
from ...services.tuning_report_service_gcp_mcp import generate_tuning_report


class TuningReportController:
    def __init__(self, app):
        # FastAPI app 인스턴스를 받아 서비스에 전달
        self.app = app

    async def create_tuning_report(self, request: TuningReport):
        try:
            result = await generate_tuning_report(request)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
