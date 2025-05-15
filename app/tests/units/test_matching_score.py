"""
매칭 점수 계산 알고리즘 테스트 모듈
이 모듈은 사용자 간 매칭 점수 계산에 사용되는 다양한 알고리즘과 함수를 단위 테스트합니다.
주요 테스트 대상:
- MBTI 기반 성격 유형 호환성 점수
- 연령대 일치 점수
- 태그(관심사, 취미 등) 일치도 계산
- 규칙 기반 유사도 계산
- 필드별 임베딩 처리 및 벡터 연산
- 최종 매칭 점수 통합 계산
"""

import json
from unittest.mock import patch

import numpy as np
import pytest
from core.matching_score import (
    age_group_match_score,
    average_field_embedding,
    compute_matching_score,
    match_tags,
    mbti_weighted_score,
    rule_based_similarity,
)


class TestMatchingScore:
    """
    매칭 점수 계산 알고리즘 테스트 클래스
    다양한 매칭 알고리즘 및 유사도 계산 함수의 정확성을 검증
    """

    def test_mbti_weighted_score(self):
        """
        MBTI 가중 점수 계산 테스트
        MBTI 성격 유형 간 호환성 점수 계산 알고리즘 검증
        각 차원(E/I, N/S, F/T, J/P)별 가중치에 따른 점수 차이 확인
        """
        # 완전 일치 케이스 - 동일한 MBTI 유형
        assert mbti_weighted_score("ESTP", "ESTP") == 1.0  # 모든 차원 일치 시 만점

        # 첫 번째 축 (E/I) 불일치 - 가중치 0.5 적용
        assert mbti_weighted_score("ESTP", "ISTP") < 1.0

        # 두 번째 축 (N/S) 불일치 - 가중치 1.0 적용
        # 인식 기능 차원은 가중치가 높아 첫 번째 축보다 점수 차이가 큼
        assert mbti_weighted_score("ESTP", "ENTP") < mbti_weighted_score("ESTP", "ISTP")

        # 세 번째 축 (F/T) 불일치 - 가중치 1.0 적용
        # 판단 기능 차원도 가중치가 높아 첫 번째 축보다 점수 차이가 큼
        assert mbti_weighted_score("ESTP", "ESFP") < mbti_weighted_score("ESTP", "ISTP")

        # 네 번째 축 (J/P) 불일치 - 가중치 0.5 적용
        assert mbti_weighted_score("ESTP", "ESTJ") < 1.0  # 생활방식 차원 불일치

        # 무효한 입력 케이스 처리 검증
        assert mbti_weighted_score("", "ESTP") == 0.0  # 빈 문자열
        assert mbti_weighted_score("ES", "ESTP") == 0.0  # 불완전한 MBTI (2글자)
        assert mbti_weighted_score("EP", "ESTP") == 0.0  # 유효하지 않은 MBTI 조합
        assert mbti_weighted_score(None, "ESTP") == 0.0  # None 값

    def test_age_group_match_score(self):
        """
        연령대 일치 점수 계산 테스트
        사용자 간 연령대 일치 여부에 따른 점수 계산 검증
        현재는 이진 점수(일치: 1.0, 불일치: 0.0) 시스템 사용
        """
        assert age_group_match_score("20대", "20대") == 1.0  # 동일 연령대 - 만점
        assert age_group_match_score("20대", "30대") == 0.0  # 다른 연령대 - 0점
        assert age_group_match_score("", "") == 1.0  # 빈 값은 동일 취급 - 만점

    def test_match_tags(self):
        """
        태그 일치도 계산 테스트
        두 사용자 간 태그(관심사, 취미 등) 일치 정도를 계산
        자카드 유사도(Jaccard similarity) 기반 계산 검증:
        일치 항목 수 / 전체 고유 항목 수
        """
        # 완전 일치 케이스
        assert match_tags(["A", "B", "C"], ["A", "B", "C"]) == 1.0  # 3/3 = 1.0

        # 부분 일치 케이스
        assert (
            match_tags(["A", "B", "C"], ["A", "B", "D"]) == 0.5
        )  # 2개 일치, 4개 고유 항목 (2/4 = 0.5)
        assert (
            match_tags(["A", "B", "C", "D", "E"], ["A", "B", "F", "G", "H", "I", "J"])
            == 0.2
        )  # 2개 일치, 10개 고유 항목 (2/10 = 0.2)

        # 완전 불일치 케이스
        assert match_tags(["A", "B", "C"], ["D", "E", "F"]) == 0.0  # 0/6 = 0

        # 빈 리스트 처리 케이스
        assert match_tags([], ["A", "B"]) == 0.0  # 빈 리스트와 비교
        assert match_tags(["A", "B"], []) == 0.0  # 빈 리스트와 비교
        assert match_tags([], []) == 0.0  # 두 리스트 모두 빈 경우

    def test_rule_based_similarity(self):
        """
        규칙 기반 유사도 계산 테스트
        MBTI, 연령대, 종교, 흡연, 음주, 성격, 선호하는 사람 특성 등을 종합적으로 고려한
        가중치 기반 유사도 계산 알고리즘 검증
        """
        # 테스트용 사용자 데이터
        user1 = {
            "MBTI": "ESTP",
            "ageGroup": "20대",
            "religion": "무교",
            "smoking": "비흡연",
            "drinking": "가끔",
            "personality": ["아담한", "듬직한"],
            "preferredPeople": ["목소리 좋은", "열정적인"],
        }

        # 자신과 동일한 사용자 (완전 일치) 케이스
        user2 = user1.copy()
        assert rule_based_similarity(user1, user2) == 1.0  # 모든 항목 일치 - 만점

        # 부분 일치 사용자 케이스
        user3 = user1.copy()
        user3["MBTI"] = "INFJ"  # 다른 MBTI (여러 차원 차이)
        user3["ageGroup"] = "30대"  # 다른 연령대
        user3["religion"] = "기독교"  # 다른 종교

        # 부분 일치 시 0~1 사이의 값이 나와야 함
        sim_score = rule_based_similarity(user1, user3)
        assert 0 < sim_score < 1.0  # 부분 일치 점수 범위 확인

    def test_average_field_embedding(self):
        """
        필드 임베딩 평균 계산 테스트
        여러 필드의 임베딩 벡터를 평균하여 통합 임베딩 생성 기능 검증
        필드 누락, 일부 필드만 선택하는 케이스 등 다양한 상황 처리 테스트
        """
        # 테스트용 임베딩 데이터 (간소화된 3차원 벡터 사용)
        field_embeddings = {
            "gender": [1.0, 2.0, 3.0],
            "MBTI": [4.0, 5.0, 6.0],
            "hobbies": [7.0, 8.0, 9.0],
        }

        # 모든 필드 평균 케이스
        avg_all = average_field_embedding(
            field_embeddings, ["gender", "MBTI", "hobbies"]
        )
        # (1+4+7)/3, (2+5+8)/3, (3+6+9)/3 = [4.0, 5.0, 6.0]
        assert avg_all == [4.0, 5.0, 6.0]

        # 일부 필드만 평균 케이스
        avg_subset = average_field_embedding(field_embeddings, ["gender", "MBTI"])
        # (1+4)/2, (2+5)/2, (3+6)/2 = [2.5, 3.5, 4.5]
        assert avg_subset == [2.5, 3.5, 4.5]

        # 존재하지 않는 필드 포함 케이스 - 존재하는 필드만 계산
        avg_missing = average_field_embedding(field_embeddings, ["gender", "unknown"])
        # gender 필드만 사용됨
        assert avg_missing == [1.0, 2.0, 3.0]

        # 모든 필드가 존재하지 않는 케이스 - 기본 빈 임베딩 반환
        avg_all_missing = average_field_embedding(
            field_embeddings, ["unknown1", "unknown2"]
        )
        # 기본 SBERT 임베딩 차원(768)의 영벡터 반환
        assert len(avg_all_missing) == 768
        assert all(v == 0.0 for v in avg_all_missing)  # 모든 값이 0.0인지 확인

    def test_compute_matching_score(self):
        """
        매칭 점수 계산 통합 테스트
        사용자 간 매칭 점수 계산의 전체 프로세스를 검증
        - 이메일 도메인 필터링 (같은 회사/조직 내 매칭)
        - 자기 자신 제외
        - 다양한 특성(MBTI, 연령대 등)을 고려한 종합 점수 계산
        """
        # 테스트용 사용자 데이터
        user_id = "1"
        user_embedding = [0.1] * 768  # 전체 임베딩
        user_meta = {
            "emailDomain": "kakaotech.com",
            "MBTI": "ESTP",
            "ageGroup": "20대",
            "religion": "무교",
            "field_embeddings": json.dumps(
                {"currentInterests": [0.2] * 768, "hobbies": [0.3] * 768}
            ),
        }

        # 테스트용 전체 사용자 데이터
        all_users = {
            "ids": ["1", "2", "3", "4"],
            "embeddings": [
                [0.1] * 768,  # user 1 (본인)
                [0.2] * 768,  # user 2
                [0.3] * 768,  # user 3
                [0.4] * 768,  # user 4 (다른 도메인)
            ],
            "metadatas": [
                {
                    "emailDomain": "kakaotech.com",
                    "MBTI": "ESTP",
                    "ageGroup": "20대",
                    "field_embeddings": json.dumps(
                        {"currentInterests": [0.2] * 768, "hobbies": [0.3] * 768}
                    ),
                },
                {
                    "emailDomain": "kakaotech.com",
                    "MBTI": "INFJ",  # 다른 MBTI
                    "ageGroup": "20대",  # 같은 연령대
                    "field_embeddings": json.dumps(
                        {"currentInterests": [0.2] * 768, "hobbies": [0.3] * 768}
                    ),
                },
                {
                    "emailDomain": "kakaotech.com",
                    "MBTI": "ENFP",  # 다른 MBTI
                    "ageGroup": "30대",  # 다른 연령대
                    "field_embeddings": json.dumps(
                        {"currentInterests": [0.5] * 768, "hobbies": [0.6] * 768}
                    ),
                },
                {
                    "emailDomain": "other.com",  # 다른 도메인 (매칭 제외 대상)
                    "MBTI": "ISTP",
                    "ageGroup": "20대",
                    "field_embeddings": json.dumps(
                        {"currentInterests": [0.7] * 768, "hobbies": [0.8] * 768}
                    ),
                },
            ],
        }

        # 함수 실행
        similarities = compute_matching_score(
            user_id, user_embedding, user_meta, all_users
        )

        # 결과 검증
        # 자신과 다른 도메인 제외, 같은 도메인 유저만 포함
        assert len(similarities) == 2
        assert "2" in similarities  # 같은 도메인, 다른 MBTI, 같은 연령대
        assert "3" in similarities  # 같은 도메인, 다른 MBTI, 다른 연령대
        assert "1" not in similarities  # 자기 자신 제외 확인
        assert "4" not in similarities  # 다른 도메인 제외 확인

        # 같은 연령대가 다른 연령대보다 유사도가 높은지 검증
        assert similarities["2"] > similarities["3"]  # 연령대 일치가 점수에 영향
