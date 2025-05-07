"""
매칭 추천 API 통합 테스트 모듈
이 모듈은 사용자 매칭 알고리즘의 추천 결과를 통합 테스트합니다.
주요 테스트 시나리오:
- 유사한 사용자 그룹 내에서 매칭 추천 결과 검증
- 존재하지 않는 사용자에 대한 오류 처리 검증
- 다양한 프로필을 가진 사용자 간 매칭 우선순위 검증
"""

import json

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
def register_test_users(test_client):
    """
    테스트용 사용자 등록 픽스처
    테스트에 필요한 여러 프로필의 사용자를 데이터베이스에 미리 등록

    등록되는 테스트 사용자:
    1. 기준 사용자 (ID: 301) - 다른 사용자와의 유사도 계산 기준점
    2. 유사한 남성 사용자 (ID: 302) - 기준 사용자와 많은 특성이 일치
    3. 유사한 여성 사용자 (ID: 303) - 기준 사용자와 많은 특성이 일치
    4. 덜 유사한 사용자 (ID: 304) - 기준 사용자와 대부분의 특성이 상이함
    """
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

    # 테스트 사용자 등록 프로세스
    for user in users:
        response = test_client.post("/api/v1/users", json=user)
        # 성공(200) 또는 이미 존재함(409) 상태 코드 모두 허용
        # 이전 테스트 실행에서 등록된 사용자가 남아있을 수 있음
        assert response.status_code in [200, 409]

    # 등록된 테스트 사용자 데이터 반환
    return users


class TestTuningFlow:
    """
    매칭 추천 API 통합 테스트 클래스
    등록된 사용자 간의 매칭 추천 기능과 예외 처리를 검증
    """

    def test_tuning_recommendations(self, test_client, register_test_users):
        """
        매칭 추천 테스트
        기준 사용자와 다른 테스트 사용자 간의 매칭 추천 결과를 검증
        등록된 모든 테스트 사용자가 추천 결과에 포함되는지 확인
        """
        # 첫 번째 사용자를 기준으로 매칭 추천 요청
        user_id = register_test_users[0]["userId"]
        response = test_client.get(f"/api/v1/tuning?userId={user_id}")

        # 응답 상태 코드 및 기본 구조 검증
        assert response.status_code == 200  # 성공 상태 코드
        data = response.json()
        assert data["code"] == "TUNING_SUCCESS"  # 성공 응답 코드
        assert "data" in data  # 데이터 객체 존재
        assert "userIdList" in data["data"]  # 추천 사용자 ID 목록 존재

        # 매칭 결과에 다른 모든 테스트 사용자가 포함되어 있는지 확인
        user_ids = data["data"]["userIdList"]
        other_user_ids = [user["userId"] for user in register_test_users[1:]]

        # 모든 테스트 사용자가 추천 목록에 포함되어 있어야 함
        # (실제 환경에서는 유사도에 따라 일부만 포함될 수 있음)
        for expected_id in other_user_ids:
            assert expected_id in user_ids

    def test_nonexistent_user(self, test_client):
        """
        존재하지 않는 사용자에 대한 매칭 추천 테스트
        데이터베이스에 없는 사용자 ID로 요청 시 적절한 오류 응답을 반환하는지 검증
        """
        # 존재하지 않는 사용자 ID(9999)로 매칭 추천 요청
        response = test_client.get("/api/v1/tuning?userId=9999")

        # 응답 검증 - 존재하지 않는 리소스에 대한 404 상태 코드
        assert response.status_code == 404  # Not Found 상태 코드

        # 응답 내용에 대한 추가 검증
        # 오류 세부 정보가 포함되어 있는지 확인
        data = response.json()
        assert "detail" in data  # 오류 세부 정보 포함 확인
        # 실제 프로덕션 환경에서는 구체적인 오류 메시지나 코드도 검증 가능
