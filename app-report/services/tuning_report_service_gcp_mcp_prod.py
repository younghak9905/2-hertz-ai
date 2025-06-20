import asyncio
import json
import os
import re
from typing import List, Set

import aiohttp
import httpx
from fastapi.responses import JSONResponse
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from openai import APIConnectionError

from ..core.enum_process import convert_to_korean
from ..core.prompt_templates.tuning_report_prompt_mcp import build_prompt
from ..models import qwen_loader_gcp_vllm
from ..schemas.tuning_schema import TuningReport, TuningReportResponse, UserProfile
from ..utils.logger import log_performance, logger

# === 재시도 관련 상수 및 예외 정의 ===

MAX_RETRY = 3


class ModelOutputValidationError(Exception):
    pass


class ChineseDetectedError(Exception):
    pass


class MissingFieldsError(Exception):
    pass


class EmptyOutputError(Exception):
    pass


RETRYABLE_EXCEPTIONS = (
    ValueError,
    MissingFieldsError,
    ChineseDetectedError,
    EmptyOutputError,
    ModelOutputValidationError,
    json.JSONDecodeError,
    aiohttp.ClientError,
    httpx.HTTPError,
    asyncio.TimeoutError,
    APIConnectionError,
)

# === 재시도 케이스 검사 ===


def contains_chinese(text: str) -> bool:
    """문자열에 중국어(한자)가 포함되어 있는지 확인"""
    return bool(re.search(r"[\u4e00-\u9fff]", text))


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
    # 이상한 연속 dash → bullet로 정리
    cleaned = re.sub(r"(\\n)+[-]+", r"\\n\\n•", cleaned)

    # 따옴표 이스케이프 처리 오염 방지
    cleaned = cleaned.replace('\\"', '"').replace("\\'", "'")

    return cleaned


def safe_json_parse(raw: str) -> dict:
    try:
        cleaned = clean_json_input(raw)
        return json.loads(cleaned)
    except Exception as e:
        raise ValueError(f"JSON 파싱 실패: {e}\n응답 원문: {repr(raw)}")


def validate_model_response(parsed: dict):
    if not parsed.get("title") or not parsed.get("content"):
        raise MissingFieldsError("title 또는 content 필드가 누락")

    text = parsed["title"] + parsed["content"]

    if contains_chinese(text):
        raise ChineseDetectedError("중국어 포함")

    if not text.strip():
        raise EmptyOutputError("빈 본문 출력")

    if (
        "```" in text
        or text.strip().startswith("{'")
        or text.strip().startswith("```json")
    ):
        raise ModelOutputValidationError("마크다운 혹은 코드블럭 포함")


# === MCP 관련 ===


def load_mcp_config():
    """현재 디렉토리의 MCP 설정 파일을 로드합니다."""
    try:
        parent_dir, _ = os.path.split(os.path.dirname(__file__))
        config_path = os.path.join(parent_dir, "config", "mcp_config.json")
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"설정 파일을 읽는 중 오류 발생: {str(e)}")
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
    if not server_config:
        logger.warning("⚠ MCP 서버 설정이 비어 있습니다. MCP 기능이 비활성화됩니다.")

    return server_config


# === 사용자 관심사 추출 ===


def extract_all_interests(user: UserProfile) -> Set[str]:
    """사용자의 모든 관심사를 추출 (정규화 적용)"""
    all_interests = set()

    # 각 필드에서 관심사 추출
    fields_to_check = [
        user.currentInterests,  # 관심사
        user.favoriteFoods,  # 좋아하는 음식
        user.likedSports,  # 좋아하는 운동
        user.pets,  # 반려동물
        user.selfDevelopment,  # 자기계발 활동
        user.hobbies,  # 취미
    ]

    for field in fields_to_check:
        if field:  # None이 아닌 경우
            for item in field:
                if item and item.strip():  # 빈 문자열이 아닌 경우
                    all_interests.add(item)

    return all_interests


def find_exact_matches(interests_a: Set[str], interests_b: Set[str]) -> List[str]:
    """정확히 일치하는 관심사 찾기"""
    return list(interests_a.intersection(interests_b))


# === MCP 검색 에이전트 ===


async def research_agent(request: TuningReport) -> str:
    try:
        server_config = create_server_config()
        client = MultiServerMCPClient(server_config)
        try:
            tools = await client.get_tools()
            logger.debug("MCP 툴 개수: ", len(tools))
        except Exception as e:
            logger.warning(f"[INFO] MCP 도구 로드 실패 또는 초기화 안됨: {e}")
            tools = []

        # 두 사용자의 정보
        userA = UserProfile(**convert_to_korean(request.userA.model_dump()))
        userB = UserProfile(**convert_to_korean(request.userB.model_dump()))

        # 2. 공통 관심사 찾기
        exact_matches = find_exact_matches(
            extract_all_interests(userA), extract_all_interests(userB)
        )
        logger.debug("공통 관심사: ", exact_matches)

        if tools:
            logger.info("🔍 1단계: 검색 전용 에이전트 실행")
            search_agent = create_react_agent(qwen_loader_gcp_vllm.get_model(), tools)

            search_messages = [
                {
                    "role": "system",
                    "content": (
                        "당신은 검색 전문가입니다. "
                        "요청된 정보를 Tavily MCP 도구로 검색한 후, 핵심 정보만 간결하게 정리해주세요.\n\n."
                        "⚠️ 중요: 검색 결과를 단순히 나열하지 말고, 공지문에서 사용할 수 있는 흥미로운 포인트들로 가공하여 제시하세요.\n\n"
                        "정리 형식:\n"
                        "1. MBTI 궁합 핵심 포인트: [한 줄 요약]\n"
                        "2. 관심사/취미 최신정보 및 트렌드\n"
                        "3. 실제 오늘의 서울 날씨 연계 스토리: [날씨를 활용한 스토리텔링 아이디어]\n"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"다음 정보들을 검색하고, 공지문 작성에 활용할 수 있도록 핵심만 정리해주세요\n"
                        f"1. {userA.MBTI}와 {userB.MBTI} MBTI 궁합 분석\n"
                        f"2. {exact_matches} 중 일부에 관한 최신 트렌드\n"
                        f"3. 서울 오늘 날씨 정보 (**섭씨 단위로 온도 기재, 화씨는 제외**)\n"
                    ),
                },
            ]

            # 검색 결과
            # search_response = await search_agent.ainvoke({"messages": search_messages})
            # 변경
            search_response = await safe_invoke_with_timeout(
                search_agent, {"messages": search_messages}, timeout=120
            )
            logger.debug(f"Search response: {search_response}")
            # 검색 결과 추출
            # TODO: 검색 성공/실패 케이스로 나눌 것
            return search_response["messages"][-1].content

        else:
            logger.info("🔄 도구 없음 - 검색 단계 생략")
            # search_results = "MCP 도구를 사용할 수 없어 검색을 수행하지 못했습니다."

    except Exception as e:
        logger.warning(f"[SEARCH ERROR] MCP 검색 실패: {type(e).__name__} - {e}")
        raise


# === 모델 응답 생성 및 검증 ===


async def safe_invoke_with_timeout(agent, payload, timeout=30):
    try:
        return await asyncio.wait_for(agent.ainvoke(payload), timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError("🔌 MCP 응답 시간 초과 (timeout)")


async def research_with_retry(request: TuningReport, max_attempts=3) -> str:
    last_error = None
    for attempt in range(max_attempts):
        try:
            return await research_agent(request)
        except Exception as e:
            error_str = str(e)
            if any(
                keyword in error_str
                for keyword in [
                    "524",
                    "Server not initialized",
                    "Bad Request",
                    "Streamable HTTP error",
                    "Timeout",
                    "Transport is closed",
                ]
            ):
                logger.warning(
                    f"[RETRYABLE-SEARCH] MCP 검색 재시도 (시도 {attempt + 1}): {type(e).__name__} - {e}"
                )
                last_error = e
                continue
            raise
    logger.error(
        f"[RETRY FAIL] 검색 {max_attempts}회 실패: {type(last_error).__name__} - {last_error}"
    )
    return "검색 결과를 불러오지 못했습니다. 기본 정보만으로 생성 진행."


async def generate_with_retry(model, messages):
    last_error = None
    for attempt in range(MAX_RETRY):
        try:
            logger.info(f"▶ 모델 응답 요청 (시도 {attempt + 1})")
            response = await model.ainvoke(messages)
            parsed = safe_json_parse(response.content)
            validate_model_response(parsed)
            return parsed
        except RETRYABLE_EXCEPTIONS as e:
            logger.warning(
                f"[RETRYABLE] 모델 응답 오류 (시도 {attempt + 1}): {type(e).__name__} - {e}"
            )
            last_error = e
            continue
        except Exception as e:
            logger.exception(f"[FATAL] 예기치 않은 오류 발생: {type(e).__name__} - {e}")
            raise

    logger.error(
        f"[RETRY FAIL] {MAX_RETRY}회 재시도 실패: {type(last_error).__name__} - {last_error}"
    )
    raise last_error


# async def tgenerate_model_response(model, messages: List[dict], attempt: int) -> dict:
#     try:
#         logger.info(f"▶ 모델 응답 요청 (시도 {attempt + 1})")
#         response = await model.ainvoke(messages)
#         parsed = safe_json_parse(response.content)
#         validate_model_response(parsed)

#         return parsed

#     except RETRYABLE_EXCEPTIONS as e:
#         logger.warning(f"[RETRYABLE] 모델 응답 오류 (시도 {attempt + 1}): {type(e).__name__} - {e}")
#         raise  # 재시도 허용

#     except Exception as e:
#         logger.exception(f"[FATAL] 예기치 않은 오류 발생 중단: {e}")
#         raise  # 치명적 예외도 상위로 전달


# === 최종 리포트 생성 ===
@log_performance(
    operation_name="generate_tuning_report", include_memory=True, include_args=True
)
async def generate_tuning_report(request: TuningReport) -> TuningReportResponse:

    try:
        logger.info(
            f"프롬프트 생성 시작 [category={request.category}, chatCount={request.chatCount}]"
        )

        # 프롬프트 생성
        search_results = await research_with_retry(request)
        logger.info("📝 2단계: 공지문 생성 전용 모델 실행")
        system_prompt = (
            "당신은 소셜 디스커버리 앱 '튜닝'의 유쾌한 감성 기자입니다.\n"
            "사용자로부터 받은 매칭유형, 채팅횟수, 프로필 정보와 아래 제공된 검색 정보를 **반드시 적극 활용**하여 흥미롭고 감정선이 살아있는 가십/연애 뉴스 스타일의 공지 기사를 작성하세요.\n\n"
            f"검색된 최신 정보: {search_results}\n"
            "⭐ 검색 정보 활용 지침:\n"
            "- 검색된 MBTI 궁합 정보를 구체적으로 언급하세요\n"
            "- 관심사 최신정보나 트렌드를 자연스럽게 녹여내세요\n"
            "- 날씨 정보를 활용한 구체적인 데이트/활동 제안을 포함하세요\n"
            "- 제공된 스토리 포인트들을 창의적으로 연결하세요\n\n"
            "📰 필수 스타일 요구사항:\n"
            "- [단독], [속보], [긴급] 중의 뉴스 헤드라인 형식 사용\n"
            "- **섹션별 이모지와 소제목**으로 가독성 향상 (예: 🧠 MBTI 분석!, 🎯 관심사 발견!)\n"
            "- **구체적인 숫자와 데이터** 언급 (대화 횟수, 온도, 통계 등)\n"
            "- **실제 트렌드나 현실적 정보** 활용 (차트, 프로그램명, 구체적 장소 등)\n"
            '- **독자 참여형 문장** 사용 ("상상 되시나요?", "궁금하시죠?")\n\n'
            "⛔ 반드시 지켜야 할 규칙:\n"
            "1. 응답은 반드시 JSON 한 개만 반환 (문자열 아님, 마크다운 아님, 설명/코드블럭 사용 금지)\n"
            '2. 출력 형식 예시: { "title": "기사 제목", "content": "공지문 본문 내용" }\n'
            "3. 줄바꿈은 반드시 문자열 안에 \\n으로 이스케이프 처리된 JSON 문자열만 출력\n"
            "4. FRIEND 유형에서는 연애나 썸 관련 표현 금지. 하트 이모지 금지. 오직 우정/의리 중심으로 작성\n"
            "5. 현실 기반 묘사로 작성 (허구적 존재: 요정, 마법, 판타지 금지)\n"
            "6. 유쾌하지만 의미를 알 수 없는 비유나 과장된 은유, 추측은 금지\n\n"
            "📝 공지문 본문 구조 (필수 5단계, 섹션별 소제목 포함):\n"
            "1. **📰 [뉴스 헤드라인] 도입부**: 익명의 두 사람 연결 상황을 속보/단독 형식으로 유쾌하게 소개\n"
            "2. **🧠 MBTI 궁합 분석**: 검색된 궁합 정보를 바탕으로 두 사람의 조합을 구체적으로 분석\n"
            "3. **🎯 공통 관심사 발견**: 관심사나 취미를 바탕으로 구체적인 활동이나 장소 제안\n"
            "4. **📱 대화 분석**: 대화 횟수를 활용해 관계 발전 단계를 분석하고 예측\n"
            "5. **🔮 마무리**: 날씨나 시의성 정보를 활용한 구체적 제안과 함께 `Stay Tuned!` 문구로 마무리\n\n"
            "📊 구체성 강화 요구사항:\n"
            "- 강조가 필요한 단어, 소제목, 포인트 문장은 **마크다운의 굵은 텍스트(`**텍스트**`)** 형식을 반드시 사용\n"
            "  - 공백 없이 별표 바로 다음에 텍스트가 와야함"
            "- 예: ❌ `** MBTI 분석 **` → ✅ `**MBTI 분석**`"
            "- 추상적 표현 대신 **구체적인 숫자, 장소, 프로그램명** 사용\n"
            "- 검색된 실제 정보를 바탕으로 **현실적인 제안** 제시\n"
            '- **"혹시", "아마도", "~일지도"** 등을 활용한 추측성 재미 요소 포함\n'
            "- 각 섹션마다 **구체적인 이모지와 소제목**으로 구조화\n\n"
        )

        userA = UserProfile(**convert_to_korean(request.userA.model_dump()))
        userB = UserProfile(**convert_to_korean(request.userB.model_dump()))
        prompt_text = build_prompt(
            category=request.category,
            chatCount=request.chatCount,
            userA=userA,
            userB=userB,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ]

        model = qwen_loader_gcp_vllm.get_model()
        # 3단계: 최대 3회 재시도하며 모델 응답 처리
        result = await generate_with_retry(model, messages)
        if result:
            title = result.get("title", "")
            content = result.get("content", "")

            logger.info(f"[SUCCESS] 튜닝 리포트 생성 완료 | title={title}")
            logger.debug(f"Final title: {title}")
            logger.debug(f"Final title: {content}")
            return TuningReportResponse(
                code="TUNING_REPORT_SUCCESS", data={"title": title, "content": content}
            )

    except Exception as e:
        logger.exception("[FAIL] 튜닝 리포트 생성 중 예외 발생")
        # 오류를 상위 계층으로 전파하여 적절한 HTTP 응답을 반환할 수 있도록 함
        return JSONResponse(
            status_code=500,
            content=TuningReportResponse(
                code="TUNING_REPORT_INTERNAL_SERVER_ERROR",
                data={"message": f"{type(e).__name__} - {e}"},
            ).model_dump(),
        )
