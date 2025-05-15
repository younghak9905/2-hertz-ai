"""
사용자 등록 API 통합 테스트 모듈
이 모듈은 사용자 등록 API의 전체 흐름을 통합 테스트합니다.
주요 테스트 시나리오:
- 사용자 등록 성공 케이스
- 중복 사용자 등록 시 오류 처리
- 잘못된 데이터 제출 시 검증 오류 처리
- 사용자 등록 후 매칭 추천 기능 연동 검증
"""

import json
import random
import time

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def test_client():
    """
    FastAPI 테스트 클라이언트 픽스처
    API 엔드포인트를 호출하기 위한 테스트용 HTTP 클라이언트 제공
    """
    return TestClient(app)


@pytest.fixture
def sample_user_data():
    """
    테스트용 사용자 등록 데이터 픽스처
    고유한 사용자 ID를 생성하여 테스트 간 충돌을 방지
    사용자 프로필에 필요한 모든 필드를 포함하는 샘플 데이터 제공
    """
    # 현재 타임스탬프와 랜덤 값을 조합하여 고유 ID 생성
    # 테스트 실행 시마다 다른 ID가 생성되어 중복 등록 문제 방지
    unique_id = int(time.time() * 1000 + random.randint(1, 1000))

    return {
        "userId": unique_id,  # 테스트용 고유 ID
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
    """
    사용자 등록 API 통합 테스트 클래스
    API 엔드포인트 호출부터 응답 검증까지 전체 흐름을 테스트
    단위 테스트와 달리 실제 API 호출을 통해 시스템 전체 동작을 검증
    """

    def test_user_registration_success(self, test_client, sample_user_data):
        """
        사용자 등록 성공 케이스 통합 테스트
        정상적인 사용자 데이터로 등록 요청 시 성공하는지 검증
        동일 ID로 중복 등록 시 409 Conflict 오류가 반환되는지 검증
        """
        # 1. 신규 사용자 등록 API 호출
        response = test_client.post("/api/v1/users", json=sample_user_data)

        # 응답 상태 코드 및 내용 검증
        assert response.status_code == 200  # 성공 상태 코드
        data = response.json()
        assert data["code"] == "EMBEDDING_REGISTER_SUCCESS"  # 성공 응답 코드

        # 2. 중복 등록 시도 (동일 ID로 재등록)
        # 이미 등록된 사용자 ID로 다시 요청하여 충돌 시나리오 테스트
        response = test_client.post("/api/v1/users", json=sample_user_data)

        # 중복 등록 시 충돌 오류 응답 검증
        assert response.status_code == 409  # Conflict 상태 코드
        data = response.json()
        assert "CONFLICT" in data["detail"]["code"]  # 충돌 오류 코드 포함 확인

    def test_user_registration_invalid_data(self, test_client):
        """
        잘못된 데이터로 사용자 등록 시도 테스트
        필수 필드가 누락된 데이터로 등록 요청 시
        422 Unprocessable Entity 오류가 반환되는지 검증
        """
        # 필수 필드 누락된 불완전한 사용자 데이터
        invalid_data = {
            "userId": 101,
            "gender": "남자",
            # 다른 필수 필드(emailDomain, MBTI 등) 누락
        }

        # 불완전한 데이터로 API 호출
        response = test_client.post("/api/v1/users", json=invalid_data)

        # 응답 검증
        assert response.status_code == 422  # Unprocessable Entity 상태 코드
        data = response.json()
        assert "detail" in data  # 검증 오류 세부 정보 포함 확인

    def test_user_registration_and_tuning(self, test_client, sample_user_data):
        """
        사용자 등록 및 매칭 추천 통합 테스트
        사용자 등록 후 해당 사용자에 대한 매칭 추천 API 호출 시
        정상적으로 매칭 결과를 반환하는지 검증
        전체 사용자 매칭 프로세스의 통합 동작 검증
        """
        # 1. 사용자 등록 단계
        response = test_client.post("/api/v1/users", json=sample_user_data)
        assert response.status_code == 200  # 등록 성공 확인

        # 2. 등록된 사용자에 대한 매칭 추천 요청
        user_id = sample_user_data["userId"]
        response = test_client.get(f"/api/v1/tuning?user_id={user_id}")

        # 응답 검증
        assert response.status_code == 200  # 성공 상태 코드
        data = response.json()
        assert "code" in data  # 응답 코드 포함 확인

        # 매칭 결과 성공 또는 매칭 없음 케이스 모두 허용
        # 테스트 환경에 따라 매칭 가능한 사용자가 없을 수 있음
        assert data["code"] in ["TUNING_SUCCESS", "TUNING_SUCCESS_BUT_NO_MATCH"]

        # 3. 매칭 결과가 있는 경우 추가 검증
        if data["code"] == "TUNING_SUCCESS":
            assert "data" in data  # 응답 데이터 포함 확인
            assert "userIdList" in data["data"]  # 매칭된 사용자 ID 목록 포함 확인
            assert isinstance(data["data"]["userIdList"], list)  # 목록 형태 확인
            # 매칭 사용자 ID 목록이 올바른 형식인지 확인
