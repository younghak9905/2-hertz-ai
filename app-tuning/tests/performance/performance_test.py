"""
성능 테스트 모듈
AI 매칭 알고리즘의 성능 개선 효과를 측정하고 분석하는 도구

주요 기능:
1. 테스트 데이터 생성
2. 실행 시간 측정 및 비교
3. 메모리 사용량 측정 및 비교
4. 매칭 품질 평가 및 분석
5. 결과 시각화 및 보고서 생성
"""

import json
import os
import time
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

# 프로젝트 모듈 import
from app.core.matching_score import compute_matching_score
from app.core.matching_score_optimized import (
    combine_embeddings,
    compute_matching_score_optimized,
)
from app.tests.load_tests.scenarios.embedding_scenarios import generate_random_user

# 결과 저장 디렉토리 설정
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def generate_test_data(num_users, same_domain_ratio=0.7, embedding_dim=768):
    """
    테스트용 사용자 데이터 생성

    Args:
        num_users: 생성할 사용자 수
        same_domain_ratio: 같은 도메인(회사)에 속한 사용자 비율
        embedding_dim: 임베딩 벡터 차원

    Returns:
        테스트용 사용자 데이터 딕셔너리 (ids, embeddings, metadatas)
    """
    # 사용자 ID 리스트 생성
    ids = [str(i) for i in range(num_users)]

    # 랜덤 임베딩 벡터 생성 (-1~1 사이의 균등 분포)
    embeddings = [
        np.random.uniform(-1, 1, embedding_dim).tolist() for _ in range(num_users)
    ]
    metadatas = []

    # 테스트용 도메인 목록
    domains = ["kakaotech.com", "other1.com", "other2.com"]

    # 각 사용자별 메타데이터 생성
    for i in range(num_users):
        # 임의의 사용자 기본 데이터 생성
        user = generate_random_user(i)

        # 같은 도메인(매칭 대상) 비율에 따라 도메인 할당
        domain_idx = 0 if i < num_users * same_domain_ratio else (i % 2) + 1
        user["emailDomain"] = domains[domain_idx]

        # 각 필드별 랜덤 임베딩 벡터 생성
        field_embeddings = {}
        for field in [
            "currentInterests",
            "hobbies",
            "likedSports",
            "favoriteFoods",
            "pets",
            "selfDevelopment",
        ]:
            field_embeddings[field] = np.random.uniform(-1, 1, embedding_dim).tolist()

        # 필드 임베딩을 JSON 문자열로 직렬화하여 저장
        user["field_embeddings"] = json.dumps(field_embeddings)
        metadatas.append(user)

    return {"ids": ids, "embeddings": embeddings, "metadatas": metadatas}


def measure_execution_time(func, *args, **kwargs):
    """
    함수 실행 시간 측정

    Args:
        func: 측정할 함수
        *args, **kwargs: 함수에 전달할, 인자

    Returns:
        (실행 결과, 실행 시간(초))
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


def run_time_comparison(user_counts, repeat=3):
    """
    여러 사용자 수에 대해 실행 시간 비교 테스트 실행

    Args:
        user_counts: 테스트할 사용자 수 목록
        repeat: 각 테스트 반복 횟수

    Returns:
        테스트 결과 데이터프레임
    """
    # 결과 저장용 딕셔너리
    results = {
        "user_count": [],
        "original_time": [],
        "optimized_time": [],
        "speedup": [],
    }

    # 각 사용자 수에 대해 테스트 실행
    for num_users in user_counts:
        print(f"테스트 중: {num_users} 사용자...")
        data = generate_test_data(num_users)

        original_times = []
        optimized_times = []

        # 각 테스트를 여러 번 반복하여 평균 계산
        for _ in range(repeat):
            # 무작위 사용자 선택 (테스트 대상)
            test_idx = np.random.randint(0, num_users)
            user_id = data["ids"][test_idx]
            user_embedding = data["embeddings"][test_idx]
            user_meta = data["metadatas"][test_idx]

            # 기존 코드 실행 시간 측정
            _, original_time = measure_execution_time(
                compute_matching_score, user_id, user_embedding, user_meta, data
            )
            original_times.append(original_time)

            # 개선된 코드 실행 시간 측정
            _, optimized_time = measure_execution_time(
                compute_matching_score_optimized,
                user_id,
                user_embedding,
                user_meta,
                data,
            )
            optimized_times.append(optimized_time)

        # 평균 시간 및 속도 향상 계산
        avg_original = np.mean(original_times)
        avg_optimized = np.mean(optimized_times)
        speedup = avg_original / avg_optimized if avg_optimized > 0 else 0

        # 결과 저장
        results["user_count"].append(num_users)
        results["original_time"].append(avg_original)
        results["optimized_time"].append(avg_optimized)
        results["speedup"].append(speedup)

        # 진행 상황 출력
        print(
            f"- 원본: {avg_original:.4f}초, 최적화: {avg_optimized:.4f}초, 속도향상: {speedup:.2f}배"
        )

    # 결과를 데이터프레임으로 변환하고 CSV로 저장
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "time_comparison_results.csv"), index=False)

    return df


def plot_time_comparison(results_df):
    """
    시간 비교 결과를 그래프로 시각화

    Args:
        results_df: 시간 비교 결과 데이터프레임
    """
    plt.figure(figsize=(12, 8))

    # 실행시간 그래프 (상단)
    plt.subplot(2, 1, 1)
    plt.plot(
        results_df["user_count"], results_df["original_time"], "o-", label="기존 코드"
    )
    plt.plot(
        results_df["user_count"],
        results_df["optimized_time"],
        "s-",
        label="개선된 코드",
    )
    plt.xlabel("사용자 수")
    plt.ylabel("실행 시간 (초)")
    plt.title("코드 개선 전후 실행 시간 비교")
    plt.legend()
    plt.grid(True)

    # 속도 향상 그래프 (하단)
    plt.subplot(2, 1, 2)
    plt.bar(results_df["user_count"], results_df["speedup"])
    plt.xlabel("사용자 수")
    plt.ylabel("속도 향상 (배)")
    plt.title("사용자 수에 따른 속도 향상")
    plt.grid(True, axis="y")

    # 그래프 레이아웃 조정 및 저장
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "time_comparison_results.png"))
    plt.close()


def measure_memory_usage(func, *args, **kwargs):
    """
    함수 실행 중 메모리 사용량 측정

    Args:
        func: 측정할 함수
        *args, **kwargs: 함수에 전달할 인자

    Returns:
        (실행 결과, 메모리 증가량(MB))
    """
    # 현재 프로세스의 메모리 사용량 확인
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)  # MB 단위로 변환

    # 함수 실행
    result = func(*args, **kwargs)

    # 실행 후 메모리 사용량 확인 및 차이 계산
    mem_after = process.memory_info().rss / (1024 * 1024)  # MB 단위로 변환
    return result, mem_after - mem_before


def test_memory_usage(func, data, test_count=10):
    """
    메모리 사용량 테스트 헬퍼 함수
    여러 번 테스트하여 평균 메모리 사용량 계산

    Args:
        func: 테스트할 함수
        data: 테스트 데이터
        test_count: 테스트 반복 횟수

    Returns:
        평균 메모리 사용량 (MB)
    """
    total_memory = 0
    # 여러 번 반복하여 평균 계산
    for _ in range(test_count):
        # 무작위 사용자 선택
        idx = np.random.randint(0, len(data["ids"]))
        user_id = data["ids"][idx]
        user_embedding = data["embeddings"][idx]
        user_meta = data["metadatas"][idx]

        # 메모리 사용량 측정
        _, memory_used = measure_memory_usage(
            func, user_id, user_embedding, user_meta, data
        )
        total_memory += memory_used

    # 평균 메모리 사용량 반환
    return total_memory / test_count


def run_memory_comparison(user_counts, test_count=5):
    """
    여러 사용자 수에 대해 메모리 사용량 비교 테스트 실행

    Args:
        user_counts: 테스트할 사용자 수 목록
        test_count: 각 테스트 반복 횟수

    Returns:
        테스트 결과 데이터프레임
    """
    # 결과 저장용 딕셔너리
    results = {
        "user_count": [],
        "original_memory": [],
        "optimized_memory": [],
        "memory_ratio": [],
    }

    # 각 사용자 수에 대한 테스트 실행
    for num_users in user_counts:
        print(f"메모리 테스트 중: {num_users} 사용자...")
        data = generate_test_data(num_users)

        # 기존 코드 메모리 사용량 측정
        original_mem = test_memory_usage(compute_matching_score, data, test_count)

        # 개선된 코드 메모리 사용량 측정
        optimized_mem = test_memory_usage(
            compute_matching_score_optimized, data, test_count
        )

        # 메모리 사용 비율 계산 (1보다 작으면 메모리 사용 감소, 1보다 크면 증가)
        memory_ratio = optimized_mem / original_mem if original_mem > 0 else 0

        # 결과 저장
        results["user_count"].append(num_users)
        results["original_memory"].append(original_mem)
        results["optimized_memory"].append(optimized_mem)
        results["memory_ratio"].append(memory_ratio)

        # 진행 상황 출력
        print(
            f"- 원본: {original_mem:.2f}MB, 최적화: {optimized_mem:.2f}MB, 비율: {memory_ratio:.2f}"
        )

    # 결과를 데이터프레임으로 변환하고 CSV로 저장
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "memory_comparison_results.csv"), index=False)

    return df


def plot_memory_comparison(results_df):
    """
    메모리 비교 결과를 그래프로 시각화

    Args:
        results_df: 메모리 비교 결과 데이터프레임
    """
    plt.figure(figsize=(12, 8))

    # 메모리 사용량 그래프 (상단)
    plt.subplot(2, 1, 1)
    plt.plot(
        results_df["user_count"], results_df["original_memory"], "o-", label="기존 코드"
    )
    plt.plot(
        results_df["user_count"],
        results_df["optimized_memory"],
        "s-",
        label="개선된 코드",
    )
    plt.xlabel("사용자 수")
    plt.ylabel("메모리 사용량 (MB)")
    plt.title("코드 개선 전후 메모리 사용량 비교")
    plt.legend()
    plt.grid(True)

    # 메모리 사용 비율 그래프 (하단)
    plt.subplot(2, 1, 2)
    plt.bar(results_df["user_count"], results_df["memory_ratio"])
    plt.axhline(y=1.0, color="r", linestyle="--")  # 기준선 (1.0)
    plt.xlabel("사용자 수")
    plt.ylabel("메모리 사용 비율 (개선/기존)")
    plt.title("사용자 수에 따른 메모리 사용 비율")
    plt.grid(True, axis="y")

    # 그래프 레이아웃 조정 및 저장
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "memory_comparison_results.png"))
    plt.close()


def compare_matching_quality(data, num_samples=20):
    """
    매칭 결과의 품질 비교
    원본 알고리즘과 개선된 알고리즘의 매칭 결과 일치도 분석

    Args:
        data: 테스트 데이터
        num_samples: 테스트 샘플 수

    Returns:
        품질 비교 결과 데이터프레임
    """
    # 결과 저장용 딕셔너리
    results = {
        "user_id": [],
        "jaccard_similarity": [],
        "rank_correlation": [],
        "top10_overlap": [],
    }

    # 여러 샘플에 대해 테스트
    for _ in range(num_samples):
        # 무작위 사용자 선택
        idx = np.random.randint(0, len(data["ids"]))
        user_id = data["ids"][idx]
        user_embedding = data["embeddings"][idx]
        user_meta = data["metadatas"][idx]

        # 두 알고리즘으로 매칭 결과 계산
        original_matches = compute_matching_score(
            user_id, user_embedding, user_meta, data
        )
        optimized_matches = compute_matching_score_optimized(
            user_id, user_embedding, user_meta, data
        )

        # 결과 집합 비교 (Jaccard 유사도)
        # - 두 알고리즘이 매칭한 사용자 ID 집합의 유사도
        orig_set = set(original_matches.keys())
        opt_set = set(optimized_matches.keys())

        # Jaccard 유사도 계산: 교집합 크기 / 합집합 크기
        jaccard = (
            len(orig_set.intersection(opt_set)) / len(orig_set.union(opt_set))
            if orig_set or opt_set
            else 1.0
        )

        # 공통 매칭 항목에 대한 순위 상관관계 계산
        common_ids = list(orig_set.intersection(opt_set))
        if len(common_ids) > 5:  # 충분한 공통 아이템이 있을 경우
            # 원본 알고리즘의 순위 계산
            orig_ranks = {
                id: i
                for i, id in enumerate(
                    sorted(common_ids, key=lambda x: original_matches[x], reverse=True)
                )
            }

            # 최적화 알고리즘의 순위 계산
            opt_ranks = {
                id: i
                for i, id in enumerate(
                    sorted(common_ids, key=lambda x: optimized_matches[x], reverse=True)
                )
            }

            # 스피어만 순위 상관계수 계산
            rank_diffs = [(orig_ranks[id] - opt_ranks[id]) ** 2 for id in common_ids]
            n = len(common_ids)
            rank_corr = 1 - (6 * sum(rank_diffs) / (n * (n**2 - 1))) if n > 1 else 1.0
        else:
            rank_corr = 1.0  # 공통 아이템이 적으면 상관관계를 1로 설정

        # 상위 10개 결과 겹침 비율 계산
        # - 두 알고리즘이 추천한 상위 10명의 사용자가 얼마나 일치하는지
        orig_top10 = set(
            sorted(
                original_matches.keys(), key=lambda x: original_matches[x], reverse=True
            )[:10]
        )
        opt_top10 = set(
            sorted(
                optimized_matches.keys(),
                key=lambda x: optimized_matches[x],
                reverse=True,
            )[:10]
        )

        # 상위 10개 항목 겹침 비율 계산
        top10_overlap = (
            len(orig_top10.intersection(opt_top10)) / 10
            if orig_top10 and opt_top10
            else 1.0
        )

        # 결과 저장
        results["user_id"].append(user_id)
        results["jaccard_similarity"].append(jaccard)
        results["rank_correlation"].append(rank_corr)
        results["top10_overlap"].append(top10_overlap)

    # 결과를 데이터프레임으로 변환하고 CSV로 저장
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "quality_comparison_results.csv"), index=False)

    return df


def analyze_quality_results(quality_results):
    """
    매칭 품질 결과 분석 및 시각화

    Args:
        quality_results: 품질 비교 결과 데이터프레임
    """
    # 요약 통계 계산
    summary = {
        "metric": ["Jaccard 유사도", "순위 상관관계", "상위 10개 겹침률"],
        "mean": [
            quality_results["jaccard_similarity"].mean(),
            quality_results["rank_correlation"].mean(),
            quality_results["top10_overlap"].mean(),
        ],
        "min": [
            quality_results["jaccard_similarity"].min(),
            quality_results["rank_correlation"].min(),
            quality_results["top10_overlap"].min(),
        ],
        "max": [
            quality_results["jaccard_similarity"].max(),
            quality_results["rank_correlation"].max(),
            quality_results["top10_overlap"].max(),
        ],
    }
    summary_df = pd.DataFrame(summary)
    print("매칭 품질 지표 요약:")
    print(summary_df)

    # 요약 통계 파일로 저장
    summary_df.to_csv(os.path.join(RESULTS_DIR, "quality_summary.csv"), index=False)

    # 품질 지표 분포 그래프 생성
    plt.figure(figsize=(12, 6))

    metrics = ["jaccard_similarity", "rank_correlation", "top10_overlap"]
    labels = ["Jaccard 유사도", "순위 상관관계", "상위 10개 겹침률"]

    # 박스플롯으로 분포 시각화
    plt.boxplot([quality_results[m] for m in metrics], labels=labels)
    plt.title("매칭 품질 지표 분포")
    plt.ylabel("점수 (0-1)")
    plt.grid(True, axis="y")

    # 그래프 저장
    plt.savefig(os.path.join(RESULTS_DIR, "quality_comparison_results.png"))
    plt.close()


def run_all_tests():
    """
    모든 성능 테스트를 순차적으로 실행
    - 실행 시간 비교
    - 메모리 사용량 비교
    - 매칭 품질 비교
    """
    print("=" * 50)
    print("매칭 알고리즘 성능 테스트 시작")
    print("=" * 50)

    # 1. 실행 시간 비교 테스트
    print("\n1. 실행 시간 비교 테스트")
    print("-" * 50)
    user_counts = [10, 50, 100, 500, 1000]  # 테스트할 사용자 수
    time_results = run_time_comparison(user_counts)
    plot_time_comparison(time_results)

    # 2. 메모리 사용량 비교 테스트
    print("\n2. 메모리 사용량 비교 테스트")
    print("-" * 50)
    memory_results = run_memory_comparison([10, 50, 100, 500, 1000])
    plot_memory_comparison(memory_results)

    # 3. 매칭 품질 비교 테스트
    print("\n3. 매칭 품질 비교 테스트")
    print("-" * 50)
    test_data = generate_test_data(2000)  # 대규모 테스트 데이터 생성
    quality_results = compare_matching_quality(test_data, num_samples=30)
    analyze_quality_results(quality_results)

    # 테스트 완료 메시지
    print("\n테스트 완료! 결과는 다음 디렉토리에 저장되었습니다:")
    print(RESULTS_DIR)


# 스크립트로 직접 실행 시 모든 테스트 실행
if __name__ == "__main__":
    run_all_tests()
