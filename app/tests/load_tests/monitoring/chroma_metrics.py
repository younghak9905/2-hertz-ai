# app/tests/load_tests/monitoring/chroma_metrics.py

import os
import time
from typing import Dict, List

import psutil

from app.core.vector_database import chroma_client


def get_chroma_collection_stats() -> Dict:
    """ChromaDB 컬렉션 통계 수집"""
    stats = {}

    # 컬렉션 정보 가져오기
    collections = chroma_client.list_collections()
    stats["collection_count"] = len(collections)

    collection_details = []
    for col in collections:
        count = col.count()
        collection_details.append({"name": col.name, "count": count})

    stats["collections"] = collection_details

    # ChromaDB 디렉토리 크기 확인
    chroma_dir = os.path.abspath("./chroma_db")  # 실제 경로로 수정
    stats["disk_usage_mb"] = get_directory_size_mb(chroma_dir)

    return stats


def get_directory_size_mb(path: str) -> float:
    """디렉토리 크기 계산 (MB)"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)  # MB로 변환


def measure_chroma_query_time(collection_name: str, num_samples: int = 10) -> Dict:
    """ChromaDB 쿼리 시간 측정"""
    collection = chroma_client.get_collection(collection_name)

    query_times = []
    start_time = time.time()

    for _ in range(num_samples):
        # 랜덤 임베딩 벡터로 쿼리
        random_embedding = [0.1] * 768  # 랜덤 벡터 (실제 테스트에서는 다양한 벡터 사용)

        query_start = time.time()
        collection.query(query_embeddings=[random_embedding], n_results=10)
        query_end = time.time()

        query_times.append(query_end - query_start)

    total_time = time.time() - start_time

    return {
        "collection": collection_name,
        "sample_count": num_samples,
        "avg_query_time": sum(query_times) / len(query_times),
        "min_query_time": min(query_times),
        "max_query_time": max(query_times),
        "total_time": total_time,
    }


def measure_chroma_performance() -> Dict:
    """ChromaDB 전반적인 성능 측정"""
    results = {}

    # 컬렉션 통계
    results["stats"] = get_chroma_collection_stats()

    # 쿼리 성능
    query_performance = {}
    for collection in ["user_profiles", "user_similarities"]:
        try:
            query_performance[collection] = measure_chroma_query_time(collection)
        except Exception as e:
            query_performance[collection] = {"error": str(e)}

    results["query_performance"] = query_performance

    # 메모리 사용량
    results["process_memory_mb"] = psutil.Process(os.getpid()).memory_info().rss / (
        1024 * 1024
    )

    return results
