# app/tests/load_tests/monitoring/visualizer.py

import json
import os
from datetime import datetime
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from app.tests.load_tests.config import TestConfig


class ResultVisualizer:
    """부하 테스트 결과 시각화"""

    def __init__(self, results_dir: str = "load_test_results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def save_results(self, metrics: Dict, config: TestConfig):
        """테스트 지표 및 그래프 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(self.results_dir, f"test_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)

        # 지표 JSON으로 저장
        with open(os.path.join(result_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # 설정 저장
        with open(os.path.join(result_dir, "config.json"), "w") as f:
            json.dump(config.__dict__, f, indent=2)

        # 그래프 생성 및 저장
        self._create_response_time_histogram(metrics, result_dir)
        self._create_throughput_graph(metrics, result_dir)
        self._create_system_resource_graph(metrics, result_dir)

        return result_dir

    def _create_response_time_histogram(self, metrics: Dict, output_dir: str):
        """응답 시간 히스토그램 생성"""
        if "response_time" not in metrics or not metrics.get("response_times", []):
            return

        plt.figure(figsize=(10, 6))
        plt.hist(metrics.get("response_times", []), bins=30, alpha=0.7, color="blue")
        plt.axvline(
            metrics["response_time"]["mean"],
            color="red",
            linestyle="dashed",
            linewidth=1,
            label=f'Mean: {metrics["response_time"]["mean"]:.3f}s',
        )
        plt.axvline(
            metrics["response_time"]["p95"],
            color="green",
            linestyle="dashed",
            linewidth=1,
            label=f'95th: {metrics["response_time"]["p95"]:.3f}s',
        )

        plt.title("Response Time Distribution")
        plt.xlabel("Response Time (seconds)")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)

        plt.savefig(os.path.join(output_dir, "response_time_histogram.png"))
        plt.close()

    def _create_throughput_graph(self, metrics: Dict, output_dir: str):
        """처리량 그래프 생성"""
        if "throughput" not in metrics:
            return

        # 단순한 막대 그래프
        plt.figure(figsize=(8, 5))
        plt.bar(["Throughput"], [metrics["throughput"]], color="blue", width=0.4)
        plt.title("Request Throughput")
        plt.ylabel("Requests per Second")
        plt.grid(True, axis="y", linestyle="--", alpha=0.7)

        plt.savefig(os.path.join(output_dir, "throughput.png"))
        plt.close()

    def _create_system_resource_graph(self, metrics: Dict, output_dir: str):
        """시스템 자원 사용량 그래프 생성"""
        if "system" not in metrics:
            return

        plt.figure(figsize=(10, 6))

        resources = ["CPU Usage (%)", "Memory Usage (%)"]
        means = [metrics["system"]["cpu_mean"], metrics["system"]["memory_mean"]]
        maxes = [metrics["system"]["cpu_max"], metrics["system"]["memory_max"]]

        x = np.arange(len(resources))
        width = 0.35

        plt.bar(x - width / 2, means, width, label="Average", color="blue")
        plt.bar(x + width / 2, maxes, width, label="Maximum", color="orange")

        plt.title("System Resource Usage")
        plt.xticks(x, resources)
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True, axis="y", linestyle="--", alpha=0.7)

        plt.savefig(os.path.join(output_dir, "system_resources.png"))
        plt.close()
