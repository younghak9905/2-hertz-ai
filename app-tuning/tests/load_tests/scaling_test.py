# app/tests/load_tests/scaling_test.py

import asyncio
import json
import os
from datetime import datetime

from app.tests.load_tests.config import SCALING_TEST_CONFIGS
from app.tests.load_tests.monitoring.metrics_collector import MetricsCollector
from app.tests.load_tests.run_load_test import run_load_test
from app.utils.logger import logger


async def run_scaling_tests():
    """사용자 수를 점진적으로 늘려가며 성능 변화 측정"""
    results = []

    # 결과 저장 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"scaling_test_results_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)

    # 각 설정으로 순차적 테스트 실행
    for idx, config in enumerate(SCALING_TEST_CONFIGS):
        logger.info(
            f"Running scaling test {idx+1}/{len(SCALING_TEST_CONFIGS)}: {config.num_users} users"
        )

        # 테스트 실행
        metrics = await run_load_test(config)

        # 결과 저장
        results.append(
            {
                "users": config.num_users,
                "concurrent": config.concurrent_requests,
                "metrics": metrics,
            }
        )

        # JSON 파일로 결과 저장
        with open(f"{result_dir}/scaling_test_{config.num_users}_users.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # 다음 테스트 전 시스템 안정화를 위한 대기
        logger.info(f"Waiting 30 seconds before next test...")
        await asyncio.sleep(30)

    # 최종 성능 변화 분석
    analyze_scaling_results(results, result_dir)

    return results


def analyze_scaling_results(results, result_dir):
    """사용자 수 증가에 따른 성능 변화 분석"""
    # 분석 데이터 추출
    users = [r["users"] for r in results]
    response_times = [r["metrics"]["response_time"]["mean"] for r in results]
    p95_times = [r["metrics"]["response_time"]["p95"] for r in results]
    throughputs = [r["metrics"]["throughput"] for r in results]
    cpu_usages = [r["metrics"]["system"]["cpu_mean"] for r in results]
    memory_usages = [r["metrics"]["system"]["memory_mean"] for r in results]

    # 결과 시각화 및 저장
    import matplotlib.pyplot as plt

    # 응답 시간 vs 사용자 수 그래프
    plt.figure(figsize=(12, 8))
    plt.plot(users, response_times, "o-", label="Mean Response Time")
    plt.plot(users, p95_times, "s-", label="P95 Response Time")
    plt.title("Response Time vs User Count")
    plt.xlabel("Number of Users")
    plt.ylabel("Response Time (seconds)")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{result_dir}/scaling_response_times.png")

    # 처리량 vs 사용자 수 그래프
    plt.figure(figsize=(12, 8))
    plt.plot(users, throughputs, "o-")
    plt.title("Throughput vs User Count")
    plt.xlabel("Number of Users")
    plt.ylabel("Requests per Second")
    plt.grid(True)
    plt.savefig(f"{result_dir}/scaling_throughput.png")

    # 리소스 사용량 vs 사용자 수 그래프
    plt.figure(figsize=(12, 8))
    plt.plot(users, cpu_usages, "o-", label="CPU Usage (%)")
    plt.plot(users, memory_usages, "s-", label="Memory Usage (%)")
    plt.title("Resource Usage vs User Count")
    plt.xlabel("Number of Users")
    plt.ylabel("Resource Usage (%)")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{result_dir}/scaling_resources.png")

    # 결과 해석 및 임계점 식별
    logger.info(f"Scaling test analysis completed and saved to {result_dir}")


if __name__ == "__main__":
    asyncio.run(run_scaling_tests())
