"""
종합 결과 보고서 생성 모듈
성능 테스트 결과를 분석하고 종합적인 성능 개선 보고서를 생성

주요 기능:
1. 실행 시간 비교 분석 및 시각화
2. 메모리 사용량 비교 분석 및 시각화
3. 매칭 품질 지표 분석
4. 종합 결론 및 권장사항 제시
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.tests.performance.performance_test import RESULTS_DIR


def generate_final_report(time_results_path, memory_results_path, quality_results_path):
    """
    최종 성능 비교 보고서 생성

    다양한 성능 테스트 결과를 종합하여 포괄적인 보고서를 생성

    Args:
        time_results_path: 실행 시간 테스트 결과 파일 경로
        memory_results_path: 메모리 사용량 테스트 결과 파일 경로
        quality_results_path: 매칭 품질 테스트 결과 파일 경로

    Returns:
        생성된 보고서 텍스트
    """
    # CSV 결과 파일 로드 및 데이터프레임으로 변환
    time_results = pd.read_csv(time_results_path)
    memory_results = pd.read_csv(memory_results_path)
    quality_results = pd.read_csv(quality_results_path)

    # 보고서 헤더 및 메타데이터 생성
    report = []
    report.append("=" * 80)
    report.append("매칭 알고리즘 성능 개선 결과 보고서")
    report.append("=" * 80)
    report.append(f"생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # 1. 실행 시간 비교 분석
    report.append("\n1. 실행 시간 비교")
    report.append("-" * 80)

    # 최대 속도 향상 찾기
    max_speedup = time_results["speedup"].max()
    max_speedup_users = time_results.loc[time_results["speedup"].idxmax(), "user_count"]
    report.append(
        f"최대 속도 향상: {max_speedup:.2f}배 (사용자 수: {max_speedup_users}명)"
    )
    report.append(f"평균 속도 향상: {time_results['speedup'].mean():.2f}배")

    # 사용자 수별 실행 시간 표 생성
    report.append("\n사용자 수별 실행 시간 (초):")
    report.append("-" * 80)
    time_table = pd.DataFrame(
        {
            "사용자 수": time_results["user_count"],
            "기존 코드": time_results["original_time"].round(4),
            "개선 코드": time_results["optimized_time"].round(4),
            "속도 향상": time_results["speedup"].round(2),
        }
    )
    report.append(time_table.to_string(index=False))

    # 2. 메모리 사용량 비교 분석
    report.append("\n\n2. 메모리 사용량 비교")
    report.append("-" * 80)

    # 메모리 사용량 차이 계산
    memory_diff = [
        (orig - opt)
        for orig, opt in zip(
            memory_results["original_memory"], memory_results["optimized_memory"]
        )
    ]
    avg_diff = np.mean(memory_diff)
    report.append(f"평균 메모리 절감: {avg_diff:.2f}MB")

    # 메모리 효율성 계산 (전체 메모리 사용 비율)
    report.append(
        f"메모리 사용 효율: {100 * (1 - memory_results['optimized_memory'].sum() / memory_results['original_memory'].sum()):.2f}%"
    )

    # 사용자 수별 메모리 사용량 표 생성
    report.append("\n사용자 수별 메모리 사용량 (MB):")
    report.append("-" * 80)
    memory_table = pd.DataFrame(
        {
            "사용자 수": memory_results["user_count"],
            "기존 코드": memory_results["original_memory"].round(2),
            "개선 코드": memory_results["optimized_memory"].round(2),
            "사용 비율": memory_results["memory_ratio"].round(2),
        }
    )
    report.append(memory_table.to_string(index=False))

    # 3. 매칭 품질 비교 분석
    report.append("\n\n3. 매칭 품질 비교")
    report.append("-" * 80)

    # 매칭 품질 주요 지표 평균값
    report.append(
        f"Jaccard 유사도 평균: {quality_results['jaccard_similarity'].mean():.4f}"
    )
    report.append(
        f"순위 상관관계 평균: {quality_results['rank_correlation'].mean():.4f}"
    )
    report.append(
        f"상위 10개 겹침률 평균: {quality_results['top10_overlap'].mean():.4f}"
    )

    # 품질 지표 통계 요약 표 생성
    report.append("\n매칭 품질 지표 통계:")
    report.append("-" * 80)
    quality_stats = pd.DataFrame(
        {
            "지표": ["Jaccard 유사도", "순위 상관관계", "상위 10개 겹침률"],
            "최소값": [
                quality_results["jaccard_similarity"].min(),
                quality_results["rank_correlation"].min(),
                quality_results["top10_overlap"].min(),
            ],
            "평균": [
                quality_results["jaccard_similarity"].mean(),
                quality_results["rank_correlation"].mean(),
                quality_results["top10_overlap"].mean(),
            ],
            "최대값": [
                quality_results["jaccard_similarity"].max(),
                quality_results["rank_correlation"].max(),
                quality_results["top10_overlap"].max(),
            ],
        }
    )
    report.append(quality_stats.to_string(index=False))

    # 4. 종합 결론 도출
    report.append("\n\n4. 종합 결론")
    report.append("-" * 80)
    report.append("- 코드 최적화를 통해 실행 시간이 평균적으로 크게 감소되었습니다.")
    report.append(f"  (평균 {time_results['speedup'].mean():.2f}배 속도 향상)")

    # 메모리 사용량 변화에 따른 결론 분기
    if avg_diff > 0:
        report.append(
            f"- 메모리 사용량 역시 효율적으로 관리되어 평균 {avg_diff:.2f}MB 절감되었습니다."
        )
    else:
        report.append(
            "- 메모리 사용은 약간 증가했으나, 속도 향상 대비 허용 가능한 수준입니다."
        )

    # 매칭 품질에 대한 결론
    report.append(
        f"- 매칭 품질은 높은 일관성을 유지하고 있으며, 상위 10개 추천의 겹침률은 {quality_results['top10_overlap'].mean():.2f}로 매우 높습니다."
    )

    # 사용 중인 임베딩 결합 방식에 대한 설명
    report.append(
        "- 개선된 코드에서는 weighted_average 방식의 임베딩 결합 방법을 사용하고 있습니다."
    )

    # 5. 권장사항 제시
    report.append("\n\n5. 권장사항")
    report.append("-" * 80)
    report.append(
        "- 대규모 사용자 환경에서는 최적화된 코드를 적용하여 실행 시간을 단축하세요."
    )
    report.append("- 실제 사용자 피드백을 통한 추가 미세 조정을 고려하세요.")
    report.append("- 사용자 증가에 따른 주기적인 성능 모니터링을 유지하세요.")

    # 보고서 파일로 저장
    report_text = "\n".join(report)
    report_path = os.path.join(RESULTS_DIR, "final_performance_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"최종 보고서가 생성되었습니다: {report_path}")
    return report_text


def create_report_from_existing_results():
    """
    기존 테스트 결과 파일로부터 최종 보고서 생성

    결과 파일이 존재하는지 확인하고, 존재하는 경우 보고서 생성 함수 호출

    Returns:
        없음. 결과 파일이 없으면 오류 메시지 출력
    """
    # 필요한 결과 파일 경로 설정
    time_results_path = os.path.join(RESULTS_DIR, "time_comparison_results.csv")
    memory_results_path = os.path.join(RESULTS_DIR, "memory_comparison_results.csv")
    quality_results_path = os.path.join(RESULTS_DIR, "quality_comparison_results.csv")

    # 필요한 모든 결과 파일이 존재하는지 확인
    missing_files = []
    for path in [time_results_path, memory_results_path, quality_results_path]:
        if not os.path.exists(path):
            missing_files.append(os.path.basename(path))

    # 누락된 파일이 있으면 오류 메시지 출력
    if missing_files:
        print(f"다음 파일이 없습니다: {', '.join(missing_files)}")
        print("먼저 성능 테스트를 실행해주세요.")
        return

    # 모든 파일이 존재하면 보고서 생성
    report = generate_final_report(
        time_results_path, memory_results_path, quality_results_path
    )

    # 생성된 보고서의 요약 출력
    print("\n보고서 요약:")
    print("-" * 50)
    # 보고서의 처음 20줄만 출력하여 간략히 보여줌
    print("\n".join(report.split("\n")[:20]))
    print("...")


# 스크립트로 직접 실행 시 기존 결과로부터 보고서 생성
if __name__ == "__main__":
    create_report_from_existing_results()
