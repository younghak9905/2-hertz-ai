# app/tests/load_tests/scenarios/tuning_scenarios.py

import asyncio
import random
import time
from typing import Dict, List, Optional, Tuple

import aiohttp
from fastapi.testclient import TestClient

from app.main import app
from app.tests.load_tests.monitoring.metrics_collector import MetricsCollector
from app.utils.logger import logger


async def request_tuning_async(
    session: aiohttp.ClientSession, base_url: str, user_id: int
) -> Optional[Dict]:
    """비동기 HTTP 클라이언트를 사용하여 튜닝 매칭 요청"""
    start_time = time.time()
    try:
        async with session.get(
            f"{base_url}/api/v1/tuning?userId={user_id}", timeout=30
        ) as response:
            # 응답 본문 읽기 (바이트 스트림 완전히 소비하여 정확한 응답 시간 측정)
            response_bytes = await response.read()

            try:
                response_data = await response.json()
            except:
                try:
                    response_text = response_bytes.decode("utf-8")
                    response_data = {"error": f"Invalid JSON: {response_text[:100]}..."}
                except:
                    response_data = {"error": "Failed to decode response"}

            elapsed = time.time() - start_time
            return {
                "status_code": response.status,
                "data": response_data,
                "user_id": user_id,
                "response_time": elapsed,  # 응답 시간 기록
            }
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"Error requesting tuning for user {user_id}: {str(e)}"
        logger.error(error_msg)
        return {
            "status_code": 500,
            "error": error_msg,
            "user_id": user_id,
            "response_time": elapsed,  # 오류 발생 시에도 응답 시간 기록
        }


# app/tests/load_tests/scenarios/tuning_scenarios.py의 run_tuning_matching_test 함수 수정


async def run_tuning_matching_test(
    num_requests: int,
    concurrent_requests: int,
    user_id_range: Tuple[int, int],
    metrics_collector: Optional[MetricsCollector] = None,
) -> List[Dict]:
    """동시 튜닝 매칭 부하 테스트 실행"""
    base_url = "http://localhost:8000"  # 테스트 서버 URL

    # 테스트할 사용자 ID 생성 (범위 내에서 랜덤 선택, 중복 허용)
    min_id, max_id = user_id_range[0], user_id_range[0] + user_id_range[1] - 1

    # 범위 제한 (존재하는 사용자 ID만 테스트)
    user_ids = [random.randint(min_id, max_id) for _ in range(num_requests)]

    results = []
    semaphore = asyncio.Semaphore(concurrent_requests)

    async def request_with_semaphore(user_id):
        start_time = time.time()
        success = False

        async with semaphore:
            result = await request_tuning_async(session, base_url, user_id)

        elapsed = time.time() - start_time

        # 성공 여부 확인
        if result and result.get("status_code") == 200:
            success = True

        # 메트릭 수집
        if metrics_collector:
            metrics_collector.add_response(success, elapsed)

        return result

    async with aiohttp.ClientSession() as session:
        # 세마포어를 사용하여 동시 요청 수 제한
        tasks = [request_with_semaphore(user_id) for user_id in user_ids]
        results = await asyncio.gather(*tasks)

    # 성공/실패 통계 계산
    success_count = sum(1 for r in results if r and r["status_code"] == 200)
    error_count = sum(1 for r in results if not r or r["status_code"] != 200)

    logger.info(
        f"Tuning matching test completed: {success_count} successful, {error_count} failed"
    )
    return [r for r in results if r]


# async def run_tuning_matching_test(
#     num_requests: int, concurrent_requests: int, user_id_range: Tuple[int, int]
# ) -> List[Dict]:
#     """동시 튜닝 매칭 부하 테스트 실행"""
#     base_url = "http://localhost:8000"  # 테스트 서버 URL

#     # 테스트할 사용자 ID 생성 (범위 내에서 랜덤 선택, 중복 허용)
#     user_ids = [
#         random.randint(user_id_range[0], user_id_range[1]) for _ in range(num_requests)
#     ]

#     results = []
#     semaphore = asyncio.Semaphore(concurrent_requests)

#     async def request_with_semaphore(user_id):
#         async with semaphore:
#             return await request_tuning_async(session, base_url, user_id)

#     async with aiohttp.ClientSession() as session:
#         # 세마포어를 사용하여 동시 요청 수 제한
#         tasks = [request_with_semaphore(user_id) for user_id in user_ids]
#         results = await asyncio.gather(*tasks)

#     # 성공/실패 통계 계산
#     success_count = sum(1 for r in results if r and r["status_code"] == 200)
#     error_count = sum(1 for r in results if not r or r["status_code"] != 200)

#     logger.info(
#         f"Tuning matching test completed: {success_count} successful, {error_count} failed"
#     )
#     return [r for r in results if r]


# 동기식 테스트 클라이언트를 사용한 테스트 (로컬 테스트용)
def request_tuning_sync(client: TestClient, user_id: int) -> Dict:
    """동기 테스트 클라이언트를 사용하여 튜닝 매칭 요청"""
    try:
        response = client.get(f"/api/v1/tuning?userId={user_id}")
        return {
            "status_code": response.status_code,
            "data": response.json(),
            "user_id": user_id,
        }
    except Exception as e:
        logger.error(f"Error requesting tuning for user {user_id}: {str(e)}")
        return {"status_code": 500, "data": {"error": str(e)}, "user_id": user_id}


def run_tuning_matching_test_sync(
    num_requests: int, user_id_range: Tuple[int, int]
) -> List[Dict]:
    """동기식 튜닝 매칭 테스트 (로컬 테스트용)"""
    client = TestClient(app)

    # 테스트할 사용자 ID 생성 (범위 내에서 랜덤 선택, 중복 허용)
    user_ids = [
        random.randint(user_id_range[0], user_id_range[1]) for _ in range(num_requests)
    ]

    results = []
    for user_id in user_ids:
        result = request_tuning_sync(client, user_id)
        results.append(result)

    # 성공/실패 통계 계산
    success_count = sum(1 for r in results if r["status_code"] == 200)
    error_count = sum(1 for r in results if r["status_code"] != 200)

    logger.info(
        f"Tuning matching test completed: {success_count} successful, {error_count} failed"
    )
    return results
