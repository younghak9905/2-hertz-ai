# 뉴스 콘텐츠 생성 서비스
import json
import logging
import re

from app.core.prompt_templates.tuning_report_prompt import build_prompt
from app.models import qwen_loader
from app.schemas.tuning_schema import TuningReport, TuningReportResponse

logger = logging.getLogger(__name__)


def clean_and_extract_response(content: str) -> dict:
    """
    LLM 응답에서 불필요한 메타 텍스트 제거 및 JSON 파싱 시도
    """
    clean_text = re.sub(r"^<[^>]+>\s*", "", content).strip()
    try:
        return json.loads(clean_text)
    except Exception:
        return {"title": "", "content": clean_text}


async def generate_tuning_report(request: TuningReport) -> TuningReportResponse:
    """
    소셜봇 채팅 메시지를 생성하는 서비스
    """
    try:
        prompt_text = build_prompt(
            category=request.category, userA=request.userA, userB=request.userB
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "당신은 데이팅 앱의 작가입니다. 주어진 사용자 정보로 흥미롭고 유쾌한 **공지문 제목과 본문만** 작성해주세요. "
                    "기사 외에는 어떤 설명이나 생각도 쓰지 마세요. 기사 내용은 마크다운 형식으로 작성하고 기사를 json 형식으로 아래처럼 구성하세요:\n\n"
                    '{ "title": "기사 제목", "content": "기사 내용" }'
                ),
            },
            {"role": "user", "content": prompt_text},
        ]

        model = qwen_loader.get_model()
        model_response = model.get_response(messages)
        print(model_response)

        # 응답 본문 전처리 및 파싱
        raw_content = model_response.get("content", "")
        parsed = clean_and_extract_response(raw_content)

        title = parsed.get("title", "")
        content = parsed.get("content", "")

        return TuningReportResponse(
            code="TUNING_REPORT_SUCCESS", data={"title": title, "content": content}
        )

    except Exception as e:
        logger.error(f"Error generating bot chat message: {str(e)}")
        raise e
