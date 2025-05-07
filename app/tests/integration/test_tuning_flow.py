# tests/integration/test_tuning_flow.py (수정 필요)
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def test_client():
    return TestClient(app)


@pytest.fixture
def register_test_users(test_client):
    """테스트용 사용자 등록"""
    users = [
        # 첫 번째 테스트 사용자 (기준)
        {
            "userId": 301,  # 고유 ID 사용
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
            "userId": 302,
            "emailDomain": "test.com",
            "gender": "남자",
            "ageGroup": "AGE_20S",
            "MBTI": "ENFP",
            "religion": "NON_RELIGIOUS",
            "smoking": "NO_SMOKING",
            "drinking": "SOMETIMES",
            "personality": ["NICE", "INTROVERTED"],
            "preferredPeople": ["NICE_VOICE"],
            "currentInterests": ["BAKING", "MOVIES"],
            "favoriteFoods": ["FRUIT", "KOREAN"],
            "likedSports": ["BOWLING"],
            "pets": ["FISH"],
            "selfDevelopment": ["READING"],
            "hobbies": ["GAMING"],
        },
        # 세 번째 테스트 사용자 (여성, 유사함)
        {
            "userId": 303,
            "emailDomain": "test.com",
            "gender": "여자",
            "ageGroup": "AGE_20S",
            "MBTI": "ENFJ",
            "religion": "NON_RELIGIOUS",
            "smoking": "NO_SMOKING",
            "drinking": "SOMETIMES",
            "personality": ["NICE_VOICE", "DOESNT_SWEAR"],
            "preferredPeople": ["KIND"],
            "currentInterests": ["DRAWING"],
            "favoriteFoods": ["WESTERN"],
            "likedSports": ["YOGA"],
            "pets": ["HAMSTER"],
            "selfDevelopment": ["READING"],
            "hobbies": ["MUSIC"],
        },
        # 네 번째 테스트 사용자 (덜 유사함)
        {
            "userId": 304,
            "emailDomain": "test.com",
            "gender": "여자",
            "ageGroup": "AGE_30S",
            "MBTI": "ISTJ",
            "religion": "CHRISTIANITY",
            "smoking": "SOMETIMES",
            "drinking": "OFTEN",
            "personality": ["ACTIVE", "PASSIONATE"],
            "preferredPeople": ["COOL", "WITTY"],
            "currentInterests": ["MAKEUP", "FORTUNE_TELLING"],
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
    def test_tuning_recommendations(self, test_client, register_test_users):
        """매칭 추천 테스트"""
        # 첫 번째 사용자를 기준으로 매칭 추천
        user_id = register_test_users[0]["userId"]
        response = test_client.get(f"/api/v1/tuning?userId={user_id}")

        # 응답 검증
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == "TUNING_SUCCESS"
        assert "data" in data
        assert "userIdList" in data["data"]

        # 매칭 결과에 다른 테스트 사용자들이 포함되어 있는지 확인
        user_ids = data["data"]["userIdList"]
        other_user_ids = [user["userId"] for user in register_test_users[1:]]

        for expected_id in other_user_ids:
            assert expected_id in user_ids

    def test_nonexistent_user(self, test_client):
        """존재하지 않는 사용자에 대한 매칭 추천 테스트"""
        response = test_client.get("/api/v1/tuning?userId=9999")

        # 응답 검증
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == "TUNING_SUCCESS_BUT_NO_MATCH"
        assert data["data"] is None
