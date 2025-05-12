"""
모든 테스트를 순차적으로 실행하는 스크립트
성능 테스트와 결과 보고서 생성을 통합적으로 실행하는 진입점

주요 기능:
1. 프로젝트 경로 설정
2. 시간 측정 및 로깅
3. 성능 테스트 실행
4. 최종 결과 보고서 생성
"""

import os
import sys
import time
from datetime import datetime

# 프로젝트 루트 디렉토리를 Python 모듈 검색 경로에 추가
# 이를 통해 app 패키지를 어디서든 import할 수 있음
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

# 테스트 및 보고서 생성 모듈 import
from app.tests.performance.generate_report import create_report_from_existing_results
from app.tests.performance.performance_test import run_all_tests


def run_complete_test():
    """
    모든 테스트를 실행하고 결과 리포트 생성

    전체 테스트 프로세스:
    1. 실행 시간, 메모리 사용량, 매칭 품질 테스트 실행
    2. 테스트 결과를 종합한 최종 보고서 생성

    Returns:
        없음. 테스트 결과는 파일로 저장됨
    """
    # 테스트 시작 시간 기록
    start_time = time.time()

    # 테스트 시작 로그 출력
    print(f"성능 테스트 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 1. 성능 테스트 실행 (시간, 메모리, 품질 비교)
    run_all_tests()

    print("\n" + "=" * 80)

    # 2. 테스트 결과를 바탕으로 최종 보고서 생성
    create_report_from_existing_results()

    # 총 실행 시간 계산 및 출력
    elapsed = time.time() - start_time
    print(f"\n모든 테스트 완료! 총 소요 시간: {elapsed/60:.2f}분")


# 스크립트로 직접 실행 시 전체 테스트 진행
if __name__ == "__main__":
    run_complete_test()
