# tests/integration/test_user_registration.py
import json

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def test_client():
    return TestClient(app)


@pytest.fixture
def sample_user_data():
    """테스트용 사용자 등록 데이터"""
    return {
        "userId": 100,  # 테스트용 고유 ID
        "emailDomain": "test.com",
        "gender": "남자",
        "ageGroup": "AGE_20S",
        "MBTI": "ESTP",
        "religion": "NON_RELIGIOUS",
        "smoking": "NO_SMOKING",
        "drinking": "SOMETIMES",
        "personality": ["KIND", "INTROVERTED"],
        "preferredPeople": ["NICE_VOICE", "DOESNT_SWEAR", "PASSIONATE"],
        "currentInterests": ["BAKING", "DRAWING", "PLANT_PARENTING"],
        "favoriteFoods": ["FRUIT", "WESTERN", "STREET_FOOD"],
        "likedSports": ["BOWLING", "BILLIARDS", "YOGA"],
        "pets": ["FISH", "HAMSTER", "RABBIT"],
        "selfDevelopment": ["READING", "STUDYING", "CAFE_STUDY"],
        "hobbies": ["GAMING", "MUSIC"],
    }


class TestUserRegistrationFlow:
    def test_user_registration_success(self, test_client, sample_user_data):
        """사용자 등록 성공 케이스 통합 테스트"""
        # API 호출
        response = test_client.post("/api/v1/users", json=sample_user_data)

        # 응답 검증
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == "EMBEDDING_REGISTER_SUCCESS"

        # 중복 등록 시도 (충돌 시나리오)
        response = test_client.post("/api/v1/users", json=sample_user_data)
        assert response.status_code == 409
        data = response.json()
        assert "CONFLICT" in data["code"]

    def test_user_registration_invalid_data(self, test_client):
        """잘못된 데이터로 사용자 등록 시도 테스트"""
        # 필수 필드 누락
        invalid_data = {
            "userId": 101,
            "gender": "남자",
            # 다른 필수 필드 누락
        }

        # API 호출
        response = test_client.post("/api/v1/users", json=invalid_data)

        # 응답 검증
        assert response.status_code == 422  # 검증 오류 상태 코드
        data = response.json()
        assert "detail" in data  # 검증 오류 세부 정보

    def test_user_registration_and_tuning(self, test_client, sample_user_data):
        """사용자 등록 및 매칭 추천 통합 테스트"""
        # 사용자 등록
        response = test_client.post("/api/v1/users", json=sample_user_data)
        assert response.status_code == 200

        # 매칭 추천 요청
        user_id = sample_user_data["userId"]
        response = test_client.get(f"/api/v1/tuning?user_id={user_id}")

        # 응답 검증
        assert response.status_code == 200
        data = response.json()
        assert "code" in data
        assert data["code"] in ["TUNING_SUCCESS", "TUNING_SUCCESS_BUT_NO_MATCH"]

        # 매칭 결과가 있는 경우 userIdList 확인
        if data["code"] == "TUNING_SUCCESS":
            assert "data" in data
            assert "userIdList" in data["data"]
            assert isinstance(data["data"]["userIdList"], list)
