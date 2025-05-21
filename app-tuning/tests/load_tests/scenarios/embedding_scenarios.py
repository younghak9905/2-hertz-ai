# app/tests/load_tests/scenarios/embedding_scenarios.py

import asyncio
import copy
import random
import time
from typing import Dict, List, Optional

import aiohttp
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import app
from app.tests.load_tests.config import DEFAULT_USER_TEMPLATE
from app.tests.load_tests.monitoring.metrics_collector import MetricsCollector
from app.utils.logger import logger

# app/tests/load_tests/scenarios/embedding_scenarios.py의 generate_random_user 함수 수정


def generate_random_user(user_id: int) -> Dict:
    """
    랜덤한 테스트 사용자 데이터 생성
    모든 필수 카테고리에 대한 정보 포함
    """
    user = copy.deepcopy(DEFAULT_USER_TEMPLATE)
    user["userId"] = user_id

    # 이메일 도메인
    email_domains = ["loadtest.com", "kakaotech.com", "test.org"]
    user["emailDomain"] = random.choice(email_domains)

    # 성별 (gender)
    genders = ["MALE", "FEMALE"]
    user["gender"] = random.choice(genders)

    # 연령대 (ageGroup)
    age_groups = ["AGE_20S", "AGE_30S", "AGE_40S", "AGE_50S", "AGE_60_PLUS"]
    user["ageGroup"] = random.choice(age_groups)

    # MBTI
    mbti_types = [
        "ISTJ",
        "ISFJ",
        "INFJ",
        "INTJ",
        "ISTP",
        "ISFP",
        "INFP",
        "INTP",
        "ESTP",
        "ESFP",
        "ENFP",
        "ENTP",
        "ESTJ",
        "ESFJ",
        "ENFJ",
        "ENTJ",
        "UNKNOWN",
    ]
    user["MBTI"] = random.choice(mbti_types)

    # 종교 (religion)
    religions = [
        "NON_RELIGIOUS",
        "CHRISTIANITY",
        "BUDDHISM",
        "CATHOLICISM",
        "WON_BUDDHISM",
        "OTHER_RELIGION",
    ]
    user["religion"] = random.choice(religions)

    # 흡연 (smoking)
    smoking_options = [
        "NO_SMOKING",
        "SOMETIMES",
        "EVERYDAY",
        "E_CIGARETTE",
        "TRYING_TO_QUIT",
    ]
    user["smoking"] = random.choice(smoking_options)

    # 음주 (drinking)
    drinking_options = [
        "NEVER",
        "ONLY_IF_NEEDED",
        "SOMETIMES",
        "OFTEN",
        "TRYING_TO_QUIT",
    ]
    user["drinking"] = random.choice(drinking_options)

    # 성격 (personality) - 최소 1개, 최대 5개
    personality_traits = [
        "CUTE",
        "RELIABLE",
        "SMILES_OFTEN",
        "DOESNT_SWEAR",
        "NICE_VOICE",
        "TALKATIVE",
        "GOOD_LISTENER",
        "ACTIVE",
        "QUIET",
        "PASSIONATE",
        "CALM",
        "WITTY",
        "POLITE",
        "SERIOUS",
        "UNIQUE",
        "FREE_SPIRITED",
        "METICULOUS",
        "SENSITIVE",
        "COOL",
        "SINCERE",
        "LOYAL",
        "OPEN_MINDED",
        "AFFECTIONATE",
        "CONSERVATIVE",
        "CONSIDERATE",
        "NEAT",
        "POSITIVE",
        "FRUGAL",
        "CHARACTERFUL",
        "HONEST",
        "PLAYFUL",
        "DILIGENT",
        "FAMILY_ORIENTED",
        "COMPETENT",
        "SELF_MANAGING",
        "RESPONSIVE",
        "WORKAHOLIC",
        "SOCIABLE",
        "LONER",
        "COMPETITIVE",
        "EMPATHETIC",
    ]
    user["personality"] = random.sample(personality_traits, random.randint(1, 5))

    # 선호하는 사람 (preferredPeople) - 최소 1개, 최대 5개
    # personality와 동일한 트레이트 목록 사용
    user["preferredPeople"] = random.sample(personality_traits, random.randint(1, 5))

    # 최근 관심사 (currentInterests) - 최소 1개, 최대 4개
    current_interests = [
        "MOVIES",
        "NETFLIX",
        "VARIETY_SHOWS",
        "HOME_CAFE",
        "CHATTING",
        "DANCE",
        "SPACE_OUT",
        "COOKING",
        "BAKING",
        "DRAWING",
        "PLANT_PARENTING",
        "INSTRUMENT",
        "PHOTOGRAPHY",
        "FORTUNE_TELLING",
        "MAKEUP",
        "NAIL_ART",
        "INTERIOR",
        "CLEANING",
        "SCUBA_DIVING",
        "SKATEBOARDING",
        "SNEAKER_COLLECTION",
        "STOCKS",
        "CRYPTO",
    ]
    user["currentInterests"] = random.sample(current_interests, random.randint(1, 4))

    # 좋아하는 음식 (favoriteFoods) - 최소 1개, 최대 4개
    favorite_foods = [
        "TTEOKBOKKI",
        "MEXICAN",
        "CHINESE",
        "JAPANESE",
        "KOREAN",
        "VEGETARIAN",
        "MEAT_LOVER",
        "FRUIT",
        "WESTERN",
        "STREET_FOOD",
        "BAKERY",
        "HAMBURGER",
        "PIZZA",
        "BRUNCH",
        "ROOT_VEGETABLES",
        "CHICKEN",
        "VIETNAMESE",
        "SEAFOOD",
        "THAI",
        "SPICY_FOOD",
    ]
    user["favoriteFoods"] = random.sample(favorite_foods, random.randint(1, 4))

    # 좋아하는 운동 (likedSports) - 최소 1개, 최대 3개
    sports = [
        "BASEBALL",
        "SOCCER",
        "HIKING",
        "RUNNING",
        "GOLF",
        "GYM",
        "PILATES",
        "HOME_TRAINING",
        "CLIMBING",
        "CYCLING",
        "BOWLING",
        "BILLIARDS",
        "YOGA",
        "TENNIS",
        "SQUASH",
        "BADMINTON",
        "BASKETBALL",
        "SURFING",
        "CROSSFIT",
        "VOLLEYBALL",
        "PINGPONG",
        "FUTSAL",
        "FISHING",
        "SKI",
        "BOXING",
        "SNOWBOARD",
        "SHOOTING",
        "JIUJITSU",
        "SWIMMING",
        "MARATHON",
    ]
    user["likedSports"] = random.sample(sports, random.randint(1, 3))

    # 반려동물 (pets) - 최소 1개, 최대 2개
    pets = [
        "DOG",
        "CAT",
        "REPTILE",
        "AMPHIBIAN",
        "BIRD",
        "FISH",
        "LIKE_BUT_NOT_HAVE",
        "HAMSTER",
        "RABBIT",
        "NONE",
        "WANT_TO_HAVE",
    ]
    user["pets"] = random.sample(pets, random.randint(1, 2))

    # 자기계발 (selfDevelopment) - 최소 1개, 최대 3개
    self_development = [
        "READING",
        "STUDYING",
        "CAFE_STUDY",
        "LICENSE_STUDY",
        "LANGUAGE_LEARNING",
        "INVESTING",
        "MIRACLE_MORNING",
        "CAREER_DEVELOPMENT",
        "DIET",
        "MINDFULNESS",
        "LIFE_OPTIMIZATION",
        "WRITING",
    ]
    user["selfDevelopment"] = random.sample(self_development, random.randint(1, 3))

    # 취미 (hobbies) - 최소 1개, 최대 4개
    hobbies = [
        "GAMING",
        "MUSIC",
        "OUTDOOR",
        "MOVIES",
        "DRAMA",
        "CHATTING",
        "SPACE_OUT",
        "APPRECIATION",
        "DANCE",
        "COOKING",
        "BAKING",
        "DRAWING",
        "PLANT_CARE",
        "INSTRUMENT",
        "PHOTOGRAPHY",
        "WEBTOON",
        "MAKEUP",
        "INTERIOR",
        "CLEANING",
        "SCUBA_DIVING",
        "COLLECTING",
        "STOCKS",
    ]
    user["hobbies"] = random.sample(hobbies, random.randint(1, 4))

    return user


async def register_user_async(
    session: aiohttp.ClientSession, base_url: str, user_data: Dict
) -> Optional[Dict]:
    """비동기 HTTP 클라이언트를 사용하여 사용자 등록"""
    start_time = time.time()

    try:
        async with session.post(f"{base_url}/api/v1/users", json=user_data) as response:
            response_bytes = await response.read()

            try:
                response_data = await response.json()
            except:
                try:
                    response_text = response_bytes.decode("utf-8")
                    response_data = {"error": f"Invalid JSON: {response_text[:100]}..."}
                except:
                    response_data = {"error": "Failed to decode response"}

            elapsed = time.time() - start_time
            return {
                "status_code": response.status,
                "data": response_data,
                "user_id": user_data["userId"],
                "response_time": elapsed,  # 응답 시간 기록
            }
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"Error registering user {user_data['userId']}: {str(e)}"
        logger.error(error_msg)
        return {
            "status_code": 500,
            "error": error_msg,
            "user_id": user_data["userId"],
            "response_time": elapsed,  # 오류 발생 시에도 응답 시간 기록
        }


# app/tests/load_tests/scenarios/embedding_scenarios.py의 run_user_registration_test 함수 수정


async def run_user_registration_test(
    num_users: int,
    concurrent_requests: int,
    base_user_id: int,
    metrics_collector: Optional[MetricsCollector] = None,
) -> List[Dict]:
    """동시 사용자 등록 부하 테스트 실행"""
    base_url = "http://localhost:8000"  # 테스트 서버 URL

    # 테스트 사용자 데이터 생성
    users = [generate_random_user(base_user_id + i) for i in range(num_users)]

    results = []
    semaphore = asyncio.Semaphore(concurrent_requests)

    async def register_with_semaphore(user):
        start_time = time.time()
        success = False

        async with semaphore:
            result = await register_user_async(session, base_url, user)

        elapsed = time.time() - start_time

        # 성공 여부 확인
        if result and result.get("status_code") == 200:
            success = True

        # 메트릭 수집
        if metrics_collector:
            metrics_collector.add_response(success, elapsed)

        return result

    async with aiohttp.ClientSession() as session:
        # 세마포어를 사용하여 동시 요청 수 제한
        tasks = [register_with_semaphore(user) for user in users]
        results = await asyncio.gather(*tasks)

    # 성공/실패 통계 계산
    success_count = sum(1 for r in results if r and r["status_code"] == 200)
    error_count = sum(1 for r in results if not r or r["status_code"] != 200)

    logger.info(
        f"User registration test completed: {success_count} successful, {error_count} failed"
    )
    return [r for r in results if r]


# async def run_user_registration_test(
#     num_users: int, concurrent_requests: int, base_user_id: int
# ) -> List[Dict]:
#     """동시 사용자 등록 부하 테스트 실행"""
#     base_url = "http://localhost:8000"  # 테스트 서버 URL

#     # 테스트 사용자 데이터 생성
#     users = [generate_random_user(base_user_id + i) for i in range(num_users)]

#     results = []
#     semaphore = asyncio.Semaphore(concurrent_requests)

#     async def register_with_semaphore(user):
#         async with semaphore:
#             return await register_user_async(session, base_url, user)

#     async with aiohttp.ClientSession() as session:
#         # 세마포어를 사용하여 동시 요청 수 제한
#         tasks = [register_with_semaphore(user) for user in users]
#         results = await asyncio.gather(*tasks)

#     # 성공/실패 통계 계산
#     success_count = sum(1 for r in results if r and r["status_code"] == 200)
#     error_count = sum(1 for r in results if not r or r["status_code"] != 200)

#     logger.info(
#         f"User registration test completed: {success_count} successful, {error_count} failed"
#     )
#     return [r for r in results if r]


# 동기식 테스트 클라이언트를 사용한 테스트 (로컬 테스트용)
def register_user_sync(client: TestClient, user_data: Dict) -> Dict:
    """동기 테스트 클라이언트를 사용하여 사용자 등록"""
    try:
        response = client.post("/api/v1/users", json=user_data)
        return {
            "status_code": response.status_code,
            "data": response.json(),
            "user_id": user_data["userId"],
        }
    except Exception as e:
        logger.error(f"Error registering user {user_data['userId']}: {str(e)}")
        return {
            "status_code": 500,
            "data": {"error": str(e)},
            "user_id": user_data["userId"],
        }


def run_user_registration_test_sync(num_users: int, base_user_id: int) -> List[Dict]:
    """동기식 사용자 등록 테스트 (로컬 테스트용)"""
    client = TestClient(app)

    # 테스트 사용자 데이터 생성
    users = [generate_random_user(base_user_id + i) for i in range(num_users)]

    results = []
    for user in users:
        result = register_user_sync(client, user)
        results.append(result)

    # 성공/실패 통계 계산
    success_count = sum(1 for r in results if r["status_code"] == 200)
    error_count = sum(1 for r in results if r["status_code"] != 200)

    logger.info(
        f"User registration test completed: {success_count} successful, {error_count} failed"
    )
    return results
