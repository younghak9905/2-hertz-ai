import json
import os
import re

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from ..core.enum_process import convert_to_korean
from ..core.prompt_templates.tuning_report_prompt import build_prompt
from ..models import qwen_loader_gcp_ollama
from ..schemas.tuning_schema import TuningReport, TuningReportResponse, UserProfile
from ..utils.logger import log_performance, logger


def clean_json_input(text: str) -> str:
    # 시작/끝 따옴표 제거
    cleaned = text.strip()
    if cleaned.startswith("'") or cleaned.startswith('"'):
        cleaned = cleaned[1:-1]

    # 마크다운/코드블럭 제거
    cleaned = re.sub(r"^```(?:json)?", "", cleaned)
    cleaned = re.sub(r"```$", "", cleaned)

    # 줄바꿈/탭 제거 (실제 줄바꿈 -> 이스케이프된 줄바꿈으로 유도)
    cleaned = cleaned.replace("\r", "").replace("\t", " ")

    # 제어 문자 제거 (ASCII 0~31 중 허용되지 않는 것들)
    cleaned = re.sub(r"[\x00-\x1f\x7f]", "", cleaned)

    return cleaned


def safe_json_parse(raw: str) -> dict:
    try:
        cleaned = clean_json_input(raw)
        return json.loads(cleaned)
    except Exception as e:
        raise ValueError(f"JSON 파싱 실패: {e}\n응답 원문: {repr(raw)}")


def load_mcp_config():
    """현재 디렉토리의 MCP 설정 파일을 로드합니다."""
    try:
        parent_dir, _ = os.path.split(os.path.dirname(__file__))
        config_path = os.path.join(parent_dir, "config", "mcp_config.json")
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"설정 파일을 읽는 중 오류 발생: {str(e)}")
        return {}


def create_server_config():
    """MCP 서버 설정을 생성합니다."""
    config = load_mcp_config()
    server_config = {}

    if config and "mcpServers" in config:
        for server_name, server_config_data in config["mcpServers"].items():
            # command가 있으면 stdio 방식
            if "command" in server_config_data:
                server_config[server_name] = {
                    "command": server_config_data.get("command"),
                    "args": server_config_data.get("args", []),
                    "transport": "stdio",
                }
            # url이 있으면 sse 방식
            elif "url" in server_config_data:
                server_config[server_name] = {
                    "url": server_config_data.get("url"),
                    "transport": "sse",
                }

    return server_config


@log_performance(
    operation_name="generate_tuning_report", include_memory=True, include_args=True
)
async def generate_tuning_report(request: TuningReport) -> TuningReportResponse:
    server_config = create_server_config()
    client = MultiServerMCPClient(server_config)
    try:
        tools = await client.get_tools()
    except Exception as e:
        logger.warning(f"MCP 도구 로드 실패, 기본 모델 사용: {e}")
        tools = []
    tools = []
    print("MCP 툴 개수: ", len(tools))  # 테스트 디버깅용  ----> 추후 삭제
    agent = create_react_agent(qwen_loader_gcp_ollama.get_model(), tools)

    try:
        logger.info(
            f"프롬프트 생성 시작 [category={request.category}, chatCount={request.chatCount}]"
        )
        userA = UserProfile(**convert_to_korean(request.userA.model_dump()))
        userB = UserProfile(**convert_to_korean(request.userB.model_dump()))
        # 프롬프트 생성
        prompt_text = build_prompt(
            category=request.category,
            chatCount=request.chatCount,
            userA=userA,
            userB=userB,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "당신은 소셜 디스커버리 앱 '튜닝'의 유쾌한 감성 기자입니다.\n"
                    "당신의 임무는 주어진 두 사람의 정보를 바탕으로만 **흥미롭고 감정선이 살아있는 공지 기사**를 작성하는 것입니다.\n"
                    "\n"
                    "⛔ 반드시 지켜야 할 규칙:\n"
                    "1. 응답은 반드시 JSON 한 개만 반환 (문자열 아님, 마크다운 아님, 설명/코드블럭 사용 금지)\n"
                    "2. 출력 형식 예시:\n"
                    '{ "title": "기사 제목", "content": "공지문 본문 내용" }\n'
                    "❗주의: 줄바꿈은 절대로 실제 `\n`이나 여러 줄로 작성하지 말고, 반드시 문자열 안에 `\\n` 으로 이스케이프 처리된 JSON 문자열만 출력하세요.\n"
                    "✅ 예시:\n"
                    '{ "title": "제목", "content": "첫 번째 힌트!\\n두 번째 힌트!\\n..." }'
                    "\n"
                    "✅ 공지문 작성 스타일:\n"
                    "- 문단마다 자연스러운 이야기 흐름이 있도록 연결하세요.\n"
                    "- 감정, 대화, 상상력을 구체적으로 드러내세요.\n"
                    "- 각 문단은 2~3문장이며, 이모지는 자연스럽게 넣되 과도하게 사용하지 마세요.\n"
                    "- FRIEND 유형에서는 연애나 썸 관련 표현 금지. 오직 우정/의리 중심으로 작성하세요.\n"
                ),
            },
            {"role": "user", "content": prompt_text},
        ]
        model_response = await agent.ainvoke({"messages": messages})
        logger.info("▶ 리포트 생성 요청 ")
        ai_message = model_response["messages"][-1]
        logger.info("▶ 모델 응답 수신 완료")
        logger.info("▶ JSON 파싱 시작")
        ai_message_content = safe_json_parse(ai_message.content)

        title = ai_message_content.get("title", "")
        content = ai_message_content.get("content", "")
        logger.info(f"[SUCCESS] 튜닝 리포트 생성 완료 | title={title}")

        # 디버깅 출력 ----> 추후 삭제
        print(f"Final title: {title}")
        print(f"Final content: {content}")

        response = TuningReportResponse(
            code="TUNING_REPORT_SUCCESS", data={"title": title, "content": content}
        )

        return response
    except Exception as e:
        logger.exception("[FAIL] 튜닝 리포트 생성 중 예외 발생")
        # 오류를 상위 계층으로 전파하여 적절한 HTTP 응답을 반환할 수 있도록 함
        raise e
