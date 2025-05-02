# tests/units/test_matching_score.py
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
    def test_mbti_weighted_score(self):
        """MBTI 가중 점수 계산 테스트"""
        # 완전 일치
        assert mbti_weighted_score("ESTP", "ESTP") == 1.0

        # 첫 번째 축 (E/I) 불일치 - 가중치 0.5
        assert mbti_weighted_score("ESTP", "ISTP") < 1.0

        # 두 번째 축 (N/S) 불일치 - 가중치 1.0
        assert mbti_weighted_score("ESTP", "ENTP") < mbti_weighted_score("ESTP", "ISTP")

        # 세 번째 축 (F/T) 불일치 - 가중치 1.0
        assert mbti_weighted_score("ESTP", "ESFP") < mbti_weighted_score("ESTP", "ISTP")

        # 네 번째 축 (J/P) 불일치 - 가중치 0.5
        assert mbti_weighted_score("ESTP", "ESTJ") < 1.0

        # 무효한 입력
        assert mbti_weighted_score("", "ESTP") == 0.0
        assert mbti_weighted_score("ES", "ESTP") == 0.0
        assert mbti_weighted_score("EP", "ESTP") == 0.0
        assert mbti_weighted_score(None, "ESTP") == 0.0

    def test_age_group_match_score(self):
        """연령대 일치 점수 계산 테스트"""
        assert age_group_match_score("20대", "20대") == 1.0
        assert age_group_match_score("20대", "30대") == 0.0
        assert age_group_match_score("", "") == 1.0  # 빈 값 동일 취급

    def test_match_tags(self):
        """태그 일치도 계산 테스트"""
        # 완전 일치
        assert match_tags(["A", "B", "C"], ["A", "B", "C"]) == 1.0

        # 부분 일치
        assert (
            match_tags(["A", "B", "C"], ["A", "B", "D"]) == 0.5
        )  # 2개 일치, 4개 고유 항목
        assert (
            match_tags(["A", "B", "C", "D", "E"], ["A", "B", "F", "G", "H", "I", "J"])
            == 0.2
        )  # 2개 일치, 10개 고유 항목

        # 불일치
        assert match_tags(["A", "B", "C"], ["D", "E", "F"]) == 0.0

        # 빈 리스트
        assert match_tags([], ["A", "B"]) == 0.0
        assert match_tags(["A", "B"], []) == 0.0
        assert match_tags([], []) == 0.0

    def test_rule_based_similarity(self):
        """규칙 기반 유사도 계산 테스트"""
        user1 = {
            "MBTI": "ESTP",
            "ageGroup": "20대",
            "religion": "무교",
            "smoking": "비흡연",
            "drinking": "가끔",
            "personality": ["아담한", "듬직한"],
            "preferredPeople": ["목소리 좋은", "열정적인"],
        }

        # 자신과 동일한 사용자 (완전 일치)
        user2 = user1.copy()
        assert rule_based_similarity(user1, user2) == 1.0

        # 부분 일치 사용자
        user3 = user1.copy()
        user3["MBTI"] = "INFJ"  # 다른 MBTI
        user3["ageGroup"] = "30대"  # 다른 연령대
        user3["religion"] = "기독교"  # 다른 종교

        sim_score = rule_based_similarity(user1, user3)
        assert 0 < sim_score < 1.0

    def test_average_field_embedding(self):
        """필드 임베딩 평균 계산 테스트"""
        # 테스트 임베딩
        field_embeddings = {
            "gender": [1.0, 2.0, 3.0],
            "MBTI": [4.0, 5.0, 6.0],
            "hobbies": [7.0, 8.0, 9.0],
        }

        # 모든 필드 평균
        # tests/units/test_matching_score.py (계속)
        avg_all = average_field_embedding(
            field_embeddings, ["gender", "MBTI", "hobbies"]
        )
        assert avg_all == [4.0, 5.0, 6.0]  # (1+4+7)/3, (2+5+8)/3, (3+6+9)/3

        # 일부 필드만 평균
        avg_subset = average_field_embedding(field_embeddings, ["gender", "MBTI"])
        assert avg_subset == [2.5, 3.5, 4.5]  # (1+4)/2, (2+5)/2, (3+6)/2

        # 존재하지 않는 필드 포함
        avg_missing = average_field_embedding(field_embeddings, ["gender", "unknown"])
        assert avg_missing == [1.0, 2.0, 3.0]  # 존재하는 필드만 평균

        # 모든 필드가 존재하지 않음
        avg_all_missing = average_field_embedding(
            field_embeddings, ["unknown1", "unknown2"]
        )
        assert len(avg_all_missing) == 768  # 기본 SBERT 임베딩 차원
        assert all(v == 0.0 for v in avg_all_missing)  # 모두 0.0

    def test_compute_matching_score(self):
        """매칭 점수 계산 통합 테스트"""
        # 테스트 데이터
        user_id = "1"
        user_embedding = [0.1] * 768
        user_meta = {
            "emailDomain": "kakaotech.com",
            "MBTI": "ESTP",
            "ageGroup": "20대",
            "religion": "무교",
            "field_embeddings": json.dumps(
                {"currentInterests": [0.2] * 768, "hobbies": [0.3] * 768}
            ),
        }

        all_users = {
            "ids": ["1", "2", "3", "4"],
            "embeddings": [
                [0.1] * 768,  # user 1 (self)
                [0.2] * 768,  # user 2
                [0.3] * 768,  # user 3
                [0.4] * 768,  # user 4
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
                    "MBTI": "INFJ",
                    "ageGroup": "20대",
                    "field_embeddings": json.dumps(
                        {"currentInterests": [0.2] * 768, "hobbies": [0.3] * 768}
                    ),
                },
                {
                    "emailDomain": "kakaotech.com",
                    "MBTI": "ENFP",
                    "ageGroup": "30대",
                    "field_embeddings": json.dumps(
                        {"currentInterests": [0.5] * 768, "hobbies": [0.6] * 768}
                    ),
                },
                {
                    "emailDomain": "other.com",  # 다른 도메인
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
        assert (
            len(similarities) == 2
        )  # 자신과 다른 도메인 제외, 같은 도메인 유저만 포함
        assert "2" in similarities
        assert "3" in similarities
        assert "1" not in similarities  # 자신 제외
        assert "4" not in similarities  # 다른 도메인 제외

        # ID 2가 ID 3보다 유사도가 높은지 검증 (같은 연령대)
        assert similarities["2"] > similarities["3"]
