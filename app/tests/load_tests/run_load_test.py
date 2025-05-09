# app/tests/load_tests/run_load_test.py

import argparse
import asyncio
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import threading
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import psutil

from app.core.vector_database import similarity_collection, user_collection
from app.tests.load_tests.config import TestConfig
from app.tests.load_tests.scenarios.embedding_scenarios import (
    run_user_registration_test,
)
from app.tests.load_tests.scenarios.tuning_scenarios import run_tuning_matching_test
from app.utils.logger import logger


@contextmanager
def timer(name):
    """작업 실행 시간 측정을 위한 컨텍스트 매니저"""
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"[LOAD TEST] {name} completed in {elapsed:.3f} seconds")


def collect_response_stats(results: List[Dict]) -> Dict:
    """응답 시간 및 성공률 등 통계 수집"""
    if not results:
        return {"error": "No results to analyze"}

    # 응답 시간 수집
    response_times = [
        r.get("response_time", 0) for r in results if r.get("response_time") is not None
    ]

    # 성공/실패 카운트
    success_count = sum(1 for r in results if r.get("status_code") == 200)
    total_count = len(results)

    # 통계 계산
    stats = {
        "total_requests": total_count,
        "successful_requests": success_count,
        "failed_requests": total_count - success_count,
        "success_rate": success_count / total_count if total_count > 0 else 0,
    }

    # 응답 시간 통계
    if response_times:
        sorted_times = sorted(response_times)
        stats["response_time"] = {
            "min": min(response_times),
            "max": max(response_times),
            "mean": sum(response_times) / len(response_times),
            "median": sorted_times[len(sorted_times) // 2],
            "p95": (
                sorted_times[int(len(sorted_times) * 0.95)]
                if len(sorted_times) >= 20
                else max(sorted_times)
            ),
            "p99": (
                sorted_times[int(len(sorted_times) * 0.99)]
                if len(sorted_times) >= 100
                else max(sorted_times)
            ),
            "std_dev": (
                sum(
                    (t - (sum(response_times) / len(response_times))) ** 2
                    for t in response_times
                )
                / len(response_times)
            )
            ** 0.5,
        }

        # 초당 처리량 계산
        total_time = sum(response_times)
        stats["throughput"] = {
            "requests_per_second": (
                len(response_times) / total_time if total_time > 0 else 0
            ),
            "avg_response_time": stats["response_time"]["mean"],
        }

    return stats


async def run_load_test(config: TestConfig):
    """설정된 구성에 따라 부하 테스트 실행"""
    logger.info(f"Starting load test with configuration: {config}")

    # 결과 저장 디렉토리
    output_dir = "load_test_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 시스템 자원 사용량 모니터링 시작
    system_metrics = {
        "cpu": [],
        "memory": [],
        "disk_read": [],
        "disk_write": [],
        "timestamps": [],
    }

    monitoring_active = True

    def monitor_system_resources():
        last_disk = psutil.disk_io_counters()
        last_time = time.time()

        while monitoring_active:
            current_time = time.time()
            current_disk = psutil.disk_io_counters()
            time_delta = current_time - last_time

            # CPU 사용률
            system_metrics["cpu"].append(psutil.cpu_percent())

            # 메모리 사용률
            mem = psutil.virtual_memory()
            system_metrics["memory"].append(mem.percent)

            # 디스크 I/O 속도 (MB/s)
            if last_disk and time_delta > 0:
                read_delta = (current_disk.read_bytes - last_disk.read_bytes) / (
                    1024 * 1024
                )
                write_delta = (current_disk.write_bytes - last_disk.write_bytes) / (
                    1024 * 1024
                )
                system_metrics["disk_read"].append(read_delta / time_delta)
                system_metrics["disk_write"].append(write_delta / time_delta)
            else:
                system_metrics["disk_read"].append(0)
                system_metrics["disk_write"].append(0)

            system_metrics["timestamps"].append(current_time - start_time)

            last_disk = current_disk
            last_time = current_time

            # 1초마다 샘플링
            time.sleep(1)

    # 모니터링 스레드 시작
    start_time = time.time()
    monitor_thread = threading.Thread(target=monitor_system_resources)
    monitor_thread.daemon = True
    monitor_thread.start()

    all_stats = {}

    try:
        # 사용자 등록 부하 테스트
        if config.run_register_test:
            with timer("User Registration Test"):
                # 테스트 실행
                register_results = await run_user_registration_test(
                    num_users=config.num_users,
                    concurrent_requests=config.concurrent_requests,
                    base_user_id=config.base_user_id,
                )

                # 결과 분석
                register_stats = collect_response_stats(register_results)
                all_stats["registration"] = register_stats

                logger.info(
                    f"Registration Test Stats: Success rate={register_stats['success_rate']:.1%}, "
                    f"Avg time={register_stats.get('response_time', {}).get('mean', 0):.3f}s"
                )

        # 튜닝 매칭 부하 테스트
        if config.run_tuning_test:
            with timer("Tuning Matching Test"):
                # 테스트 실행
                tuning_results = await run_tuning_matching_test(
                    num_requests=config.num_requests,
                    concurrent_requests=config.concurrent_requests,
                    user_id_range=(config.base_user_id, config.num_users),
                )

                # 결과 분석
                tuning_stats = collect_response_stats(tuning_results)
                all_stats["tuning"] = tuning_stats

                logger.info(
                    f"Tuning Test Stats: Success rate={tuning_stats['success_rate']:.1%}, "
                    f"Avg time={tuning_stats.get('response_time', {}).get('mean', 0):.3f}s"
                )

        # DB 상태 확인
        try:
            user_count = len(user_collection.get().get("ids", []))
            similarity_count = len(similarity_collection.get().get("ids", []))
            logger.info(
                f"DB State - Users: {user_count}, Similarities: {similarity_count}"
            )

            all_stats["db_state"] = {
                "user_count": user_count,
                "similarity_count": similarity_count,
            }
        except Exception as e:
            logger.error(f"Error getting DB state: {str(e)}")

        # 결과 파일로 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(output_dir, f"test_results_{timestamp}.json")

        with open(result_file, "w") as f:
            json.dump(all_stats, f, indent=2)

        logger.info(f"Test results saved to {result_file}")

        # 간단한 그래프 생성 (옵션)
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            if config.run_register_test and "registration" in all_stats:
                plt.figure(figsize=(10, 6))
                rt_data = all_stats["registration"].get("response_time", {})
                if rt_data:
                    plt.bar(
                        ["Min", "Mean", "Median", "95th", "Max"],
                        [
                            rt_data.get("min", 0),
                            rt_data.get("mean", 0),
                            rt_data.get("median", 0),
                            rt_data.get("p95", 0),
                            rt_data.get("max", 0),
                        ],
                    )
                    plt.title("Registration API Response Times")
                    plt.ylabel("Time (seconds)")
                    plt.savefig(
                        os.path.join(output_dir, f"registration_times_{timestamp}.png")
                    )

            if config.run_tuning_test and "tuning" in all_stats:
                plt.figure(figsize=(10, 6))
                rt_data = all_stats["tuning"].get("response_time", {})
                if rt_data:
                    plt.bar(
                        ["Min", "Mean", "Median", "95th", "Max"],
                        [
                            rt_data.get("min", 0),
                            rt_data.get("mean", 0),
                            rt_data.get("median", 0),
                            rt_data.get("p95", 0),
                            rt_data.get("max", 0),
                        ],
                    )
                    plt.title("Tuning API Response Times")
                    plt.ylabel("Time (seconds)")
                    plt.savefig(
                        os.path.join(output_dir, f"tuning_times_{timestamp}.png")
                    )

            # 처리량 비교 (두 테스트 모두 실행한 경우)
            if config.run_register_test and config.run_tuning_test:
                plt.figure(figsize=(8, 5))
                reg_throughput = all_stats["registration"].get("throughput", 0)
                tun_throughput = all_stats["tuning"].get("throughput", 0)

                plt.bar(["Registration", "Tuning"], [reg_throughput, tun_throughput])
                plt.title("API Throughput Comparison")
                plt.ylabel("Requests per Second")
                plt.savefig(
                    os.path.join(output_dir, f"throughput_comparison_{timestamp}.png")
                )

        except Exception as e:
            logger.error(f"Error generating graphs: {str(e)}")

        # 모니터링 중지
        monitoring_active = False
        monitor_thread.join(timeout=1.0)

        # 시스템 자원 사용량 통계 계산
        if system_metrics["cpu"]:
            all_stats["system_resources"] = {
                "cpu": {
                    "mean": round(
                        sum(system_metrics["cpu"]) / len(system_metrics["cpu"]), 2
                    ),
                    "max": round(max(system_metrics["cpu"]), 2),
                    "samples": len(system_metrics["cpu"]),
                },
                "memory": {
                    "mean": round(
                        sum(system_metrics["memory"]) / len(system_metrics["memory"]), 2
                    ),
                    "max": round(max(system_metrics["memory"]), 2),
                    "samples": len(system_metrics["memory"]),
                },
                "disk_read": {
                    "mean": round(
                        sum(system_metrics["disk_read"])
                        / len(system_metrics["disk_read"]),
                        2,
                    ),
                    "max": round(max(system_metrics["disk_read"]), 2),
                    "samples": len(system_metrics["disk_read"]),
                },
                "disk_write": {
                    "mean": round(
                        sum(system_metrics["disk_write"])
                        / len(system_metrics["disk_write"]),
                        2,
                    ),
                    "max": round(max(system_metrics["disk_write"]), 2),
                    "samples": len(system_metrics["disk_write"]),
                },
            }

            # 시스템 자원 사용량 그래프 생성
            if len(system_metrics["timestamps"]) > 1:
                plt.figure(figsize=(12, 10))

                # CPU 사용률 그래프
                plt.subplot(3, 1, 1)
                plt.plot(system_metrics["timestamps"], system_metrics["cpu"], "r-")
                plt.title("CPU Usage")
                plt.ylabel("CPU (%)")
                plt.grid(True)

                # 메모리 사용률 그래프
                plt.subplot(3, 1, 2)
                plt.plot(system_metrics["timestamps"], system_metrics["memory"], "b-")
                plt.title("Memory Usage")
                plt.ylabel("Memory (%)")
                plt.grid(True)

                # 디스크 I/O 그래프
                plt.subplot(3, 1, 3)
                plt.plot(
                    system_metrics["timestamps"],
                    system_metrics["disk_read"],
                    "g-",
                    label="Read",
                )
                plt.plot(
                    system_metrics["timestamps"],
                    system_metrics["disk_write"],
                    "m-",
                    label="Write",
                )
                plt.title("Disk I/O")
                plt.ylabel("MB/s")
                plt.xlabel("Time (seconds)")
                plt.legend()
                plt.grid(True)

                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_dir, f"system_resources_{timestamp}.png")
                )
                plt.close()

                logger.info(
                    f"System resource monitoring graphs saved to {output_dir}/system_resources_{timestamp}.png"
                )

        # 전체 결과 로깅
        logger.info(
            f"CPU Usage: Avg {all_stats.get('system_resources', {}).get('cpu', {}).get('mean', 'N/A')}%, "
            f"Max {all_stats.get('system_resources', {}).get('cpu', {}).get('max', 'N/A')}%"
        )
        logger.info(
            f"Memory Usage: Avg {all_stats.get('system_resources', {}).get('memory', {}).get('mean', 'N/A')}%, "
            f"Max {all_stats.get('system_resources', {}).get('memory', {}).get('max', 'N/A')}%"
        )

        return all_stats

    except Exception as e:
        logger.error(f"Load test failed: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        raise


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description="Run load tests for TUNING API")
    parser.add_argument(
        "--users", type=int, default=100, help="Number of users to register"
    )
    parser.add_argument(
        "--requests", type=int, default=200, help="Number of tuning requests"
    )
    parser.add_argument(
        "--concurrent", type=int, default=10, help="Number of concurrent requests"
    )
    parser.add_argument(
        "--base-id", type=int, default=10000, help="Base user ID for testing"
    )
    parser.add_argument(
        "--register", action="store_true", help="Run user registration test"
    )
    parser.add_argument(
        "--tuning", action="store_true", help="Run tuning matching test"
    )
    parser.add_argument("--all", action="store_true", help="Run all tests")

    args = parser.parse_args()

    # --all 플래그가 설정되면 모든 테스트 활성화
    if args.all:
        args.register = True
        args.tuning = True

    # 최소한 하나의 테스트는 활성화되어야 함
    if not any([args.register, args.tuning]):
        parser.error(
            "At least one test must be selected (--register, --tuning, or --all)"
        )

    return args


if __name__ == "__main__":
    args = parse_args()

    config = TestConfig(
        num_users=args.users,
        num_requests=args.requests,
        concurrent_requests=args.concurrent,
        base_user_id=args.base_id,
        run_register_test=args.register,
        run_tuning_test=args.tuning,
    )

    asyncio.run(run_load_test(config))
