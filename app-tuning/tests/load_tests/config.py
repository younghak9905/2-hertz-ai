# app/tests/load_tests/config.py

from dataclasses import dataclass


@dataclass
class TestConfig:
    """부하 테스트 구성 설정"""

    num_users: int = 100  # 테스트에 사용할 사용자 수
    num_requests: int = 200  # 총 요청 수
    concurrent_requests: int = 10  # 동시 요청 수
    base_user_id: int = 10000  # 테스트 사용자 ID 시작 번호
    run_register_test: bool = True  # 사용자 등록 테스트 실행 여부
    run_tuning_test: bool = True  # 튜닝 매칭 테스트 실행 여부


# 테스트용 기본 사용자 템플릿
DEFAULT_USER_TEMPLATE = {
    "userId": 0,  # 실행 시 오버라이드됨
    "emailDomain": "loadtest.com",
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

LARGE_SCALE_CONFIG = TestConfig(
    num_users=1000,
    num_requests=2000,
    concurrent_requests=50,
    base_user_id=20000,
    run_register_test=True,
    run_tuning_test=True,
)

# 단계적 테스트 설정 추가
SCALING_TEST_CONFIGS = [
    TestConfig(num_users=10, concurrent_requests=5, base_user_id=30000),
    TestConfig(num_users=100, concurrent_requests=10, base_user_id=31000),
    TestConfig(num_users=500, concurrent_requests=20, base_user_id=32000),
    TestConfig(num_users=1000, concurrent_requests=50, base_user_id=33000),
]
