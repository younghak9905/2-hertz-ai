# 매칭 추천 api 흐름 테스트
# tests/integration/test_tuning_flow.py
import json

import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def test_client():
    return TestClient(app)


@pytest.fixture
def register_test_users(test_client):
    """테스트용 사용자 등록"""
    users = [
        # 첫 번째 테스트 사용자 (기준)
        {
            "userId": 201,
            "emailDomain": "test.com",
            "gender": "남자",
            "ageGroup": "AGE_20S",
            "MBTI": "ESTP",
            "religion": "NON_RELIGIOUS",
            "smoking": "NO_SMOKING",
            "drinking": "SOMETIMES",
            "personality": ["KIND", "INTROVERTED"],
            "preferredPeople": ["NICE_VOICE", "DOESNT_SWEAR"],
            "currentInterests": ["BAKING", "DRAWING"],
            "favoriteFoods": ["FRUIT", "WESTERN"],
            "likedSports": ["BOWLING", "YOGA"],
            "pets": ["FISH"],
            "selfDevelopment": ["READING"],
            "hobbies": ["GAMING", "MUSIC"],
        },
        # 두 번째 테스트 사용자 (남성, 유사함)
        {
            "userId": 202,
            "emailDomain": "test.com",
            "gender": "남자",
            "ageGroup": "AGE_20S",
            "MBTI": "ENFP",  # 다른 MBTI
            "religion": "NON_RELIGIOUS",
            "smoking": "NO_SMOKING",
            "drinking": "SOMETIMES",
            "personality": ["NICE", "INTROVERTED"],
            "preferredPeople": ["NICE_VOICE"],
            "currentInterests": ["BAKING", "MOVIES"],  # 일부 공통
            "favoriteFoods": ["FRUIT", "KOREAN"],
            "likedSports": ["BOWLING"],
            "pets": ["FISH"],
            "selfDevelopment": ["READING"],
            "hobbies": ["GAMING"],
        },
        # 세 번째 테스트 사용자 (여성, 유사함)
        {
            "userId": 203,
            "emailDomain": "test.com",
            "gender": "여자",
            "ageGroup": "AGE_20S",
            "MBTI": "ENFJ",  # 다른 MBTI
            "religion": "NON_RELIGIOUS",
            "smoking": "NO_SMOKING",
            "drinking": "SOMETIMES",
            "personality": ["NICE_VOICE", "DOESNT_SWEAR"],  # 선호하는 상대 성향과 일치
            "preferredPeople": ["KIND"],  # 첫 번째 사용자의 성향과 일치
            "currentInterests": ["DRAWING"],
            "favoriteFoods": ["WESTERN"],
            "likedSports": ["YOGA"],
            "pets": ["HAMSTER"],
            "selfDevelopment": ["READING"],
            "hobbies": ["MUSIC"],
        },
        # 네 번째 테스트 사용자 (덜 유사함)
        {
            "userId": 204,
            "emailDomain": "test.com",
            "gender": "여자",
            "ageGroup": "AGE_30S",  # 다른 연령대
            "MBTI": "ISTJ",  # 다른 MBTI
            "religion": "CHRISTIANITY",  # 다른 종교
            "smoking": "SOMETIMES",  # 다른 흡연 상태
            "drinking": "OFTEN",  # 다른 음주 상태
            "personality": ["ACTIVE", "PASSIONATE"],
            "preferredPeople": ["COOL", "WITTY"],
            "currentInterests": ["MAKEUP", "FORTUNE_TELLING"],  # 완전히 다른 관심사
            "favoriteFoods": ["KOREAN", "CHINESE"],
            "likedSports": ["TENNIS", "RUNNING"],
            "pets": ["DOG", "CAT"],
            "selfDevelopment": ["DIET", "LANGUAGE_LEARNING"],
            "hobbies": ["OUTDOOR", "PHOTOGRAPHY"],
        },
    ]

    # 사용자 등록
    for user in users:
        response = test_client.post("/api/v1/users", json=user)
        assert response.status_code in [200, 409]  # 성공 또는 이미 존재함

    return users


class TestTuningFlow:
    def test_tuning_all_category(self, test_client, register_test_users):
        """전체 카테고리 매칭 추천 테스트"""
        # 첫 번째 사용자를 기준으로 매칭 추천
        user_id = register_test_users[0]["userId"]
        response = test_client.get(f"/api/v1/tuning?user_id={user_id}")

        # 응답 검증
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == "TUNING_SUCCESS"
        assert "data" in data
        assert "userIdList" in data["data"]

        # 매칭 결과 확인
        user_ids = data["data"]["userIdList"]
        assert len(user_ids) > 0

        # 202, 203 사용자가 204 사용자보다 상위에 있는지 확인 (더 유사함)
        if 202 in user_ids and 204 in user_ids:
            assert user_ids.index(202) < user_ids.index(204)
        if 203 in user_ids and 204 in user_ids:
            assert user_ids.index(203) < user_ids.index(204)

    def test_tuning_opposite_gender(self, test_client, register_test_users):
        """이성 매칭 추천 테스트"""
        # 첫 번째 사용자(남성)를 기준으로 이성 매칭 추천
        user_id = register_test_users[0]["userId"]
        response = test_client.get(
            f"/api/v1/tuning?user_id={user_id}&category=opposite"
        )

        # 응답 검증
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == "TUNING_SUCCESS"
        assert "data" in data
        assert "userIdList" in data["data"]

        # 모든 추천 대상이 여성인지 확인
        user_ids = data["data"]["userIdList"]
        assert len(user_ids) > 0
        assert all(uid in [203, 204] for uid in user_ids)  # 여성 사용자만 포함
        assert 202 not in user_ids  # 남성 사용자 제외

    def test_tuning_same_gender(self, test_client, register_test_users):
        """동성 매칭 추천 테스트"""
        # 첫 번째 사용자(남성)를 기준으로 동성 매칭 추천
        user_id = register_test_users[0]["userId"]
        response = test_client.get(f"/api/v1/tuning?user_id={user_id}&category=same")

        # 응답 검증
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == "TUNING_SUCCESS"
        assert "data" in data
        assert "userIdList" in data["data"]

        # 모든 추천 대상이 남성인지 확인
        user_ids = data["data"]["userIdList"]
        assert len(user_ids) > 0
        assert all(uid in [202] for uid in user_ids)  # 남성 사용자만 포함
        assert 203 not in user_ids  # 여성 사용자 제외
        assert 204 not in user_ids  # 여성 사용자 제외

    def test_nonexistent_user(self, test_client):
        """존재하지 않는 사용자에 대한 매칭 추천 테스트"""
        response = test_client.get("/api/v1/tuning?user_id=9999")

        # 응답 검증
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == "TUNING_SUCCESS_BUT_NO_MATCH"
        assert data["data"] is None
