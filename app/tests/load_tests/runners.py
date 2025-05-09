# app/tests/load_tests/runners.py

import asyncio
import time
from typing import Dict, List, Optional

from app.tests.load_tests.config import TestConfig
from app.tests.load_tests.monitoring.metrics_collector import MetricsCollector
from app.tests.load_tests.scenarios.embedding_scenarios import (
    run_user_registration_test,
)
from app.tests.load_tests.scenarios.tuning_scenarios import run_tuning_matching_test
from app.utils.logger import logger


class TestRunner:
    """
    부하 테스트 실행을 관리하는 클래스
    다양한 테스트 패턴 지원 (순차, 병렬, 램프업 등)
    """

    def __init__(self, config: TestConfig):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.results = {}

    async def run_sequential_test(self) -> Dict:
        """
        순차적 테스트 실행 (등록 → 매칭)
        """
        logger.info(f"Starting sequential test with config: {self.config}")
        await self.metrics_collector.start_sampling()

        all_results = {}

        try:
            # 사용자 등록 테스트 실행
            if self.config.run_register_test:
                start_time = time.time()
                register_results = await run_user_registration_test(
                    num_users=self.config.num_users,
                    concurrent_requests=self.config.concurrent_requests,
                    base_user_id=self.config.base_user_id,
                )
                elapsed = time.time() - start_time

                all_results["registration"] = {
                    "results": register_results,
                    "time_taken": elapsed,
                    "success_rate": (
                        sum(1 for r in register_results if r["status_code"] == 200)
                        / len(register_results)
                        if register_results
                        else 0
                    ),
                }

            # 잠시 대기 (시스템 안정화)
            await asyncio.sleep(2)

            # 튜닝 매칭 테스트 실행
            if self.config.run_tuning_test:
                start_time = time.time()
                tuning_results = await run_tuning_matching_test(
                    num_requests=self.config.num_requests,
                    concurrent_requests=self.config.concurrent_requests,
                    user_id_range=(
                        self.config.base_user_id,
                        self.config.base_user_id + self.config.num_users - 1,
                    ),
                )
                elapsed = time.time() - start_time

                all_results["tuning"] = {
                    "results": tuning_results,
                    "time_taken": elapsed,
                    "success_rate": (
                        sum(1 for r in tuning_results if r["status_code"] == 200)
                        / len(tuning_results)
                        if tuning_results
                        else 0
                    ),
                }

            metrics = await self.metrics_collector.stop_sampling()
            all_results["metrics"] = metrics

            return all_results

        except Exception as e:
            logger.error(f"Error during sequential test: {str(e)}")
            await self.metrics_collector.stop_sampling()
            raise

    async def run_ramp_up_test(self, steps: int = 4) -> Dict:
        """
        점진적으로 부하를 증가시키는 램프업 테스트
        """
        logger.info(f"Starting ramp-up test with {steps} steps")

        results_by_step = {}

        # 각 단계별 동시 요청 수 설정
        concurrent_by_step = []
        max_concurrent = self.config.concurrent_requests

        for step in range(1, steps + 1):
            concurrent = max(1, int(max_concurrent * (step / steps)))
            concurrent_by_step.append(concurrent)

        # 각 단계별 테스트 실행
        for step, concurrent in enumerate(concurrent_by_step, 1):
            logger.info(
                f"Running step {step}/{steps} with {concurrent} concurrent requests"
            )

            # 현재 단계 설정으로 테스트 실행
            step_config = TestConfig(
                num_users=self.config.num_users,
                num_requests=self.config.num_requests // steps,  # 요청 수 분배
                concurrent_requests=concurrent,
                base_user_id=self.config.base_user_id
                + ((step - 1) * self.config.num_users // steps),
                run_register_test=self.config.run_register_test,
                run_tuning_test=self.config.run_tuning_test,
            )

            # 현재 단계 러너 생성 및 실행
            step_runner = TestRunner(step_config)
            step_results = await step_runner.run_sequential_test()

            results_by_step[f"step_{step}"] = {
                "concurrent": concurrent,
                "results": step_results,
            }

            # 다음 단계 사이에 잠시 대기
            if step < steps:
                logger.info(f"Waiting 5 seconds before next step...")
                await asyncio.sleep(5)

        return {"test_type": "ramp_up", "steps": steps, "results": results_by_step}

    async def run_stress_test(self, duration_seconds: int = 60) -> Dict:
        """
        지정 시간 동안 지속적으로 요청을 보내는 스트레스 테스트
        """
        logger.info(f"Starting stress test for {duration_seconds} seconds")
        await self.metrics_collector.start_sampling()

        start_time = time.time()
        end_time = start_time + duration_seconds

        results = []
        tasks = []

        try:
            # 테스트 선택
            test_func = None
            if self.config.run_register_test:
                test_func = run_user_registration_test
            elif self.config.run_tuning_test:
                test_func = run_tuning_matching_test
            else:
                raise ValueError("No test selected for stress test")

            # 지정 시간 동안 테스트 실행
            batch_size = self.config.concurrent_requests
            batch_number = 0

            while time.time() < end_time:
                logger.info(f"Starting batch {batch_number + 1}")

                if self.config.run_register_test:
                    # 사용자 등록 테스트
                    batch_user_id = self.config.base_user_id + (
                        batch_number * batch_size
                    )
                    batch_task = asyncio.create_task(
                        run_user_registration_test(
                            num_users=batch_size,
                            concurrent_requests=batch_size,
                            base_user_id=batch_user_id,
                        )
                    )
                else:
                    # 튜닝 매칭 테스트
                    batch_task = asyncio.create_task(
                        run_tuning_matching_test(
                            num_requests=batch_size,
                            concurrent_requests=batch_size,
                            user_id_range=(
                                self.config.base_user_id,
                                self.config.base_user_id + self.config.num_users - 1,
                            ),
                        )
                    )

                tasks.append(batch_task)
                batch_number += 1

                # 부하 조절을 위한 잠시 대기
                await asyncio.sleep(1)

            # 모든 실행 중인 태스크 완료 대기
            for task in tasks:
                batch_result = await task
                results.extend(batch_result)

            # 결과 수집
            metrics = await self.metrics_collector.stop_sampling()

            return {
                "test_type": "stress",
                "duration": duration_seconds,
                "total_batches": batch_number,
                "results": results,
                "metrics": metrics,
            }

        except Exception as e:
            logger.error(f"Error during stress test: {str(e)}")
            await self.metrics_collector.stop_sampling()
            raise

    async def run_spike_test(self, spike_users: int = 500) -> Dict:
        """
        갑작스러운 대규모 요청을 보내는 스파이크 테스트
        """
        logger.info(f"Starting spike test with {spike_users} users")
        await self.metrics_collector.start_sampling()

        try:
            # 일반 부하 테스트 실행
            normal_results = await run_user_registration_test(
                num_users=self.config.num_users,
                concurrent_requests=self.config.concurrent_requests,
                base_user_id=self.config.base_user_id,
            )

            # 잠시 대기
            logger.info("Normal load completed, waiting 5 seconds before spike...")
            await asyncio.sleep(5)

            # 스파이크 부하 테스트 실행
            spike_results = await run_user_registration_test(
                num_users=spike_users,
                concurrent_requests=spike_users,  # 모든 요청을 동시에 발생
                base_user_id=self.config.base_user_id + self.config.num_users,
            )

            metrics = await self.metrics_collector.stop_sampling()

            return {
                "test_type": "spike",
                "normal_load": self.config.num_users,
                "spike_load": spike_users,
                "normal_results": normal_results,
                "spike_results": spike_results,
                "metrics": metrics,
            }

        except Exception as e:
            logger.error(f"Error during spike test: {str(e)}")
            await self.metrics_collector.stop_sampling()
            raise
