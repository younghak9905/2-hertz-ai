import json
import logging

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from models.qwen_loader_gcp_vllm import get_model
from schemas.tuning_schema import TuningReport, TuningReportResponse

logger = logging.getLogger(__name__)


def load_mcp_config():
    """현재 디렉토리의 MCP 설정 파일을 로드합니다."""
    try:
        with open("app-report/config/mcp_config.json", "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"설정 파일을 읽는 중 오류 발생: {str(e)}")
        return None


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


async def generate_tuning_report(request: TuningReport) -> TuningReportResponse:
    server_config = create_server_config()
    client = MultiServerMCPClient(server_config)
    tools = await client.get_tools()
    agent = create_react_agent(get_model(), tools)

    try:
        # 프롬프트 생성
        prompt_text = build_prompt(
            category=request.category, userA=request.userA, userB=request.userB
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "당신은 데이팅 앱의 작가입니다. 주어진 사용자 정보로 흥미롭고 유쾌한 **공지문 제목과 본문만** 작성해주세요. "
                    "기사 외에는 어떤 설명이나 생각도 쓰지 마세요. 기사 내용은 마크다운 형식으로 작성하고 "
                    "기사를 json 형식으로 아래처럼 구성하세요:\n\n"
                    '{ "title": "기사 제목", "content": "기사 내용" }'
                ),
            },
            {"role": "user", "content": prompt_text},
        ]
        response = await agent.ainvoke({"messages": messages})
        print("response: ", response)

        return response
    except Exception as e:
        logger.error(f"Error generating tuning report: {str(e)}")
        # 오류를 상위 계층으로 전파하여 적절한 HTTP 응답을 반환할 수 있도록 함
        raise e
