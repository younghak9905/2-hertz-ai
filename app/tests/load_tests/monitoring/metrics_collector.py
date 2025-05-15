# app/tests/load_tests/monitoring/metrics_collector.py

import time
import psutil
import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TestMetrics:
    """부하 테스트 중 수집한 성능 지표"""
    # 테스트 식별 정보
    test_name: str = "unknown"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    # 응답 시간 지표
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    response_times: List[float] = field(default_factory=list)
    
    # 시스템 자원 지표
    cpu_usage_samples: List[float] = field(default_factory=list)
    memory_usage_samples: List[float] = field(default_factory=list)
    disk_io_samples: List[Dict] = field(default_factory=list)
    
    # ChromaDB 지표 
    db_operations: Dict[str, List[float]] = field(default_factory=dict)
    collection_sizes: Dict[str, List[int]] = field(default_factory=dict)
    
    def start(self):
        """테스트 시작 시간 기록"""
        self.start_time = time.time()
    
    def stop(self):
        """테스트 종료 시간 기록"""
        self.end_time = time.time()
    
    def add_response(self, success: bool, response_time: float):
        """응답 결과 기록"""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        self.response_times.append(response_time)
    
    def sample_system_metrics(self):
        """시스템 자원 사용량 샘플링"""
        # CPU 사용률
        self.cpu_usage_samples.append(psutil.cpu_percent())
        
        # 메모리 사용량 (MB)
        memory = psutil.virtual_memory()
        self.memory_usage_samples.append(memory.used / (1024 * 1024))
        
        # 디스크 I/O 정보
        disk_io = psutil.disk_io_counters()
        if disk_io:
            self.disk_io_samples.append({
                "read_bytes": disk_io.read_bytes,
                "write_bytes": disk_io.write_bytes,
                "timestamp": time.time()
            })
    
    def sample_db_metrics(self, user_collection, similarity_collection):
        """ChromaDB 상태 샘플링"""
        try:
            user_count = len(user_collection.get()["ids"])
            similarity_count = len(similarity_collection.get()["ids"])
            
            if "user_profiles" not in self.collection_sizes:
                self.collection_sizes["user_profiles"] = []
            if "user_similarities" not in self.collection_sizes:
                self.collection_sizes["user_similarities"] = []
            
            self.collection_sizes["user_profiles"].append(user_count)
            self.collection_sizes["user_similarities"].append(similarity_count)
        except Exception as e:
            print(f"Error sampling DB metrics: {str(e)}")
    
    def calculate_statistics(self) -> Dict:
        """수집된 지표로부터 통계 계산"""
        if not self.response_times:
            return {"error": "No response times collected"}
        
        response_times = sorted(self.response_times)
        
        # 기본 통계 계산
        stats = {
            "test_name": self.test_name,
            "requests": {
                "total": self.total_requests,
                "success": self.successful_requests,
                "failed": self.failed_requests,
                "success_rate": self.successful_requests / self.total_requests if self.total_requests > 0 else 0
            },
            "response_time": {
                "min": min(response_times) if response_times else 0,
                "max": max(response_times) if response_times else 0,
                "mean": sum(response_times) / len(response_times) if response_times else 0,
                "median": response_times[len(response_times) // 2] if response_times else 0,
                "p95": response_times[int(len(response_times) * 0.95)] if len(response_times) > 20 else (max(response_times) if response_times else 0),
                "p99": response_times[int(len(response_times) * 0.99)] if len(response_times) > 100 else (max(response_times) if response_times else 0)
            },
            "system": {
                "cpu_mean": sum(self.cpu_usage_samples) / len(self.cpu_usage_samples) if self.cpu_usage_samples else 0,
                "memory_mean_mb": sum(self.memory_usage_samples) / len(self.memory_usage_samples) if self.memory_usage_samples else 0,
                "cpu_max": max(self.cpu_usage_samples) if self.cpu_usage_samples else 0,
                "memory_max_mb": max(self.memory_usage_samples) if self.memory_usage_samples else 0
            },
            "database": {
                "final_user_count": self.collection_sizes["user_profiles"][-1] if self.collection_sizes.get("user_profiles") else 0,
                "final_similarity_count": self.collection_sizes["user_similarities"][-1] if self.collection_sizes.get("user_similarities") else 0
            }
        }
        
        # 처리량(throughput) 계산
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
            stats["duration"] = duration
            stats["throughput"] = {
                "requests_per_second": self.total_requests / duration if duration > 0 else 0,
                "successful_requests_per_second": self.successful_requests / duration if duration > 0 else 0
            }
        
        # 디스크 I/O 통계 추가
        if len(self.disk_io_samples) >= 2:
            first = self.disk_io_samples[0]
            last = self.disk_io_samples[-1]
            time_diff = last["timestamp"] - first["timestamp"]
            
            if time_diff > 0:
                read_diff = last["read_bytes"] - first["read_bytes"]
                write_diff = last["write_bytes"] - first["write_bytes"]
                
                stats["disk_io"] = {
                    "read_bytes_per_second": read_diff / time_diff,
                    "write_bytes_per_second": write_diff / time_diff,
                    "total_read_bytes": read_diff,
                    "total_write_bytes": write_diff
                }
        
        return stats
    
    def generate_report(self, output_dir="load_test_results"):
        """테스트 결과 보고서 생성 및 그래프 저장"""
        if not self.response_times:
            return {"error": "No data to generate report"}
        
        # 결과 디렉토리 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(output_dir, f"{self.test_name}_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)
        
        # 통계 계산
        stats = self.calculate_statistics()
        
        # JSON으로 저장
        with open(os.path.join(result_dir, "metrics.json"), "w") as f:
            json.dump(stats, f, indent=2)
        
        # 그래프 생성
        self._generate_response_time_graphs(result_dir)
        self._generate_system_resource_graphs(result_dir)
        self._generate_throughput_graph(result_dir, stats)
        
        return {
            "report_dir": result_dir,
            "stats": stats
        }
    
    def _generate_response_time_graphs(self, output_dir):
        """응답 시간 관련 그래프 생성"""
        if not self.response_times:
            return
        
        # 1. 응답 시간 히스토그램
        plt.figure(figsize=(10, 6))
        plt.hist(self.response_times, bins=min(30, len(self.response_times) // 2 + 1), alpha=0.7, color='blue')
        
        # 주요 통계치 표시
        mean_time = sum(self.response_times) / len(self.response_times)
        sorted_times = sorted(self.response_times)
        p95 = sorted_times[int(len(sorted_times) * 0.95)] if len(sorted_times) > 20 else max(sorted_times)
        
        plt.axvline(mean_time, color='red', linestyle='dashed', linewidth=1, label=f'평균: {mean_time:.3f}s')
        plt.axvline(p95, color='green', linestyle='dashed', linewidth=1, label=f'95th 백분위: {p95:.3f}s')
        
        plt.title('응답 시간 분포')
        plt.xlabel('응답 시간 (초)')
        plt.ylabel('요청 수')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.savefig(os.path.join(output_dir, "response_time_histogram.png"))
        plt.close()
        
        # 2. 응답 시간 추이 (시간순)
        if len(self.response_times) > 1:
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(self.response_times)), self.response_times, 'b-', marker='o', markersize=3)
            plt.axhline(mean_time, color='red', linestyle='--', label=f'평균: {mean_time:.3f}s')
            
            plt.title('응답 시간 추이')
            plt.xlabel('요청 순서')
            plt.ylabel('응답 시간 (초)')
            plt.legend()
            plt.grid(True)
            
            plt.savefig(os.path.join(output_dir, "response_time_trend.png"))
            plt.close()
    
    def _generate_system_resource_graphs(self, output_dir):
        """시스템 자원 사용량 그래프 생성"""
        if not self.cpu_usage_samples or not self.memory_usage_samples:
            return
        
        # CPU 및 메모리 사용량 추이
        plt.figure(figsize=(12, 8))
        
        # 두 개의 서브플롯 생성
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # CPU 사용률
        x = range(len(self.cpu_usage_samples))
        ax1.plot(x, self.cpu_usage_samples, 'r-', label='CPU 사용률')
        ax1.set_title('CPU 사용률 추이')
        ax1.set_ylabel('사용률 (%)')
        ax1.grid(True)
        ax1.legend()
        
        # 메모리 사용량 (MB)
        ax2.plot(x, self.memory_usage_samples, 'b-', label='메모리 사용량')
        ax2.set_title('메모리 사용량 추이')
        ax2.set_xlabel('샘플 순서')
        ax2.set_ylabel('사용량 (MB)')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "system_resources.png"))
        plt.close()
        
        # 평균 자원 사용량 막대 그래프
        plt.figure(figsize=(10, 6))
        
        resources = ['CPU 사용률 (%)', '메모리 사용량 (MB)']
        means = [
            sum(self.cpu_usage_samples) / len(self.cpu_usage_samples) if self.cpu_usage_samples else 0,
            sum(self.memory_usage_samples) / len(self.memory_usage_samples) if self.memory_usage_samples else 0
        ]
        maxes = [
            max(self.cpu_usage_samples) if self.cpu_usage_samples else 0,
            max(self.memory_usage_samples) if self.memory_usage_samples else 0
        ]
        
        x = np.arange(len(resources))
        width = 0.35
        
        plt.bar(x - width/2, means, width, label='평균', color='blue')
        plt.bar(x + width/2, maxes, width, label='최대', color='orange')
        
        plt.title('시스템 자원 사용량 요약')
        plt.xticks(x, resources)
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.savefig(os.path.join(output_dir, "resource_summary.png"))
        plt.close()
    
    def _generate_throughput_graph(self, output_dir, stats):
        """처리량 그래프 생성"""
        if "throughput" not in stats:
            return
        
        # 처리량 막대 그래프
        plt.figure(figsize=(8, 6))
        
        metrics = ['전체 요청', '성공한 요청']
        values = [
            stats["throughput"]["requests_per_second"],
            stats["throughput"]["successful_requests_per_second"]
        ]
        
        plt.bar(metrics, values, color=['blue', 'green'])
        plt.title('초당 처리량')
        plt.ylabel('요청/초')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # 수치 표시
        for i, v in enumerate(values):
            plt.text(i, v + 0.1, f"{v:.2f}", ha='center')
        
        plt.savefig(os.path.join(output_dir, "throughput.png"))
        plt.close()


class MetricsCollector:
    """부하 테스트 중 지표 수집기"""
    def __init__(self, test_name="unknown", sample_interval: float = 1.0):
        self.metrics = TestMetrics(test_name=test_name)
        self.sample_interval = sample_interval
        self._sampling_task = None
        self.output_dir = "load_test_results"
        os.makedirs(self.output_dir, exist_ok=True)
    
    async def start_sampling(self, user_collection=None, similarity_collection=None):
        """백그라운드 샘플링 작업 시작"""
        import asyncio
        
        self.metrics.start()
        
        # 샘플링 루프 실행
        async def _sample_loop():
            try:
                while True:
                    self.metrics.sample_system_metrics()
                    if user_collection and similarity_collection:
                        self.metrics.sample_db_metrics(user_collection, similarity_collection)
                    await asyncio.sleep(self.sample_interval)
            except asyncio.CancelledError:
                pass
        
        self._sampling_task = asyncio.create_task(_sample_loop())
    
    async def stop_sampling(self) -> Dict:
        """샘플링 중지 및 결과 반환"""
        import asyncio
        
        if self._sampling_task:
            self._sampling_task.cancel()
            try:
                await self._sampling_task
            except asyncio.CancelledError:
                pass
        
        self.metrics.stop()
        report = self.metrics.generate_report(self.output_dir)
        return report["stats"]
    
    def add_response(self, success: bool, response_time: float):
        """응답 추가"""
        self.metrics.add_response(success, response_time)