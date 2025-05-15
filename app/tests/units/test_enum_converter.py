"""
열거형(Enum) 변환 유틸리티 테스트 모듈
이 모듈은 영문 코드값을 한글 표시값으로 변환하는 기능을 테스트합니다.
주요 테스트 대상:
- 단일 필드 변환 (성별, 연령대, 종교 등)
- 리스트 필드 변환 (성격, 취미 등)
- 알 수 없는 값 처리
- 다양한 필드 조합 변환
- Enum 매핑 테이블 검증
"""

import pytest
from core.enum_process import ENUM_MAPPINGS, convert_to_korean


def test_convert_single_fields():
    """
    단일 필드 변환 테스트
    개별 사용자 속성(성별, 연령대, 종교 등)의 영문 코드값이
    올바른 한글 표시값으로 변환되는지 검증
    """
    # 테스트 입력 데이터 - 단일 값을 가진 필드들
    test_data = {
        "userId": 1,  # 변환 대상 아님
        "gender": "MALE",  # 성별
        "ageGroup": "AGE_20S",  # 연령대
        "religion": "NON_RELIGIOUS",  # 종교
        "smoking": "NO_SMOKING",  # 흡연 여부
        "drinking": "SOMETIMES",  # 음주 빈도
    }

    # 기대 결과 - 한글로 변환된 값
    expected = {
        "userId": 1,  # 변환 대상이 아닌 필드는 그대로 유지
        "gender": "남자",
        "ageGroup": "20대",
        "religion": "무교",
        "smoking": "비흡연",
        "drinking": "가끔",
    }

    # 함수 실행 및 결과 출력
    result = convert_to_korean(test_data)
    print("Result of test1: ", result)

    # 전체 결과가 기대값과 일치하는지 검증
    assert result == expected


def test_convert_list_fields():
    """
    리스트 필드 변환 테스트
    배열 형태의 태그(성격, 취미 등)가 각각
    올바른 한글 표시값으로 변환되는지 검증
    """
    # 테스트 입력 데이터 - 리스트 값을 가진 필드들
    test_data = {"personality": ["CUTE", "RELIABLE"], "hobbies": ["GAMING", "MUSIC"]}

    # 기대 결과 - 리스트의 각 항목이 한글로 변환됨
    expected = {"personality": ["아담한", "듬직한"], "hobbies": ["게임", "음악"]}

    # 함수 실행 및 결과 출력
    result = convert_to_korean(test_data)
    print("Result of test2: ", result)

    # 전체 결과가 기대값과 일치하는지 검증
    assert result == expected


def test_handle_unknown_values():
    """
    알 수 없는 값 처리 테스트
    매핑 테이블에 존재하지 않는 코드값이 입력될 경우
    원본 값이 그대로 유지되는지 검증
    """
    # 테스트 입력 데이터 - 알 수 없는 값 포함
    test_data = {"gender": "UNKNOWN", "personality": ["UNKNOWN", "RELIABLE"]}

    # 기대 결과 - 알 수 없는 값은 원본 유지, 알려진 값은 변환
    expected = {"gender": "UNKNOWN", "personality": ["UNKNOWN", "듬직한"]}

    # 함수 실행 및 결과 출력
    result = convert_to_korean(test_data)
    print("Result of test3: ", result)

    # 전체 결과가 기대값과 일치하는지 검증
    assert result == expected


def test_convert_multiple_fields():
    """
    다양한 필드 조합 변환 테스트
    단일 값 필드와 리스트 필드가 혼합된 복잡한 데이터 구조에서
    모든 필드가 올바르게 변환되는지 검증
    """
    # 테스트 입력 데이터 - 단일 값과 리스트 값이 혼합된 복잡한 구조
    test_data = {
        "userId": 1,  # 변환 대상 아님
        "gender": "MALE",
        "ageGroup": "AGE_20S",
        "religion": "NON_RELIGIOUS",
        "smoking": "NO_SMOKING",
        "drinking": "SOMETIMES",
        "personality": ["CUTE", "RELIABLE"],
        "hobbies": ["GAMING", "MUSIC"],
        "likedSports": ["TENNIS", "YOGA"],
    }

    # 함수 실행 및 결과 출력
    result = convert_to_korean(test_data)
    print("Result of test4: ", result)

    # 단일 값 필드 검증 - 각 필드별로 개별 검증
    assert result["gender"] == "남자"
    assert result["ageGroup"] == "20대"
    assert result["religion"] == "무교"
    assert result["smoking"] == "비흡연"
    assert result["drinking"] == "가끔"

    # 리스트 필드 검증 - 각 리스트의 항목별 포함 여부 검증
    assert "아담한" in result["personality"]
    assert "듬직한" in result["personality"]
    assert "게임" in result["hobbies"]
    assert "음악" in result["hobbies"]
    assert "테니스" in result["likedSports"]
    assert "요가" in result["likedSports"]


def test_convert_partial_fields():
    """
    일부 필드만 포함된 경우 변환 테스트
    일부 필드만 존재하는 경우에도 변환 함수가
    정상적으로 동작하는지 검증
    """
    # 테스트 입력 데이터 - 일부 필드만 포함
    test_data = {
        "userId": 1,  # 변환 대상 아님
        "gender": "MALE",
        "preferredPeople": ["NICE_VOICE", "PASSIONATE"],
    }

    # 함수 실행 및 결과 출력
    result = convert_to_korean(test_data)
    print("Result of test5: ", result)

    # 개별 필드 검증
    assert result["gender"] == "남자"
    assert "목소리 좋은" in result["preferredPeople"]
    assert "열정적인" in result["preferredPeople"]


def test_enum_mapping_coverage():
    """
    Enum 매핑 테이블 검증
    필수 필드에 대한 매핑 테이블이 모두 정의되어 있는지
    일부 대표 매핑 값이 올바르게 정의되어 있는지 검증
    """
    # 필수 필드 매핑 테이블 존재 여부 검증
    assert "ageGroup" in ENUM_MAPPINGS
    assert "gender" in ENUM_MAPPINGS
    assert "religion" in ENUM_MAPPINGS
    assert "smoking" in ENUM_MAPPINGS
    assert "drinking" in ENUM_MAPPINGS
    assert "personality" in ENUM_MAPPINGS
    assert "preferredPeople" in ENUM_MAPPINGS
    assert "currentInterests" in ENUM_MAPPINGS
    assert "favoriteFoods" in ENUM_MAPPINGS
    assert "likedSports" in ENUM_MAPPINGS
    assert "pets" in ENUM_MAPPINGS
    assert "selfDevelopment" in ENUM_MAPPINGS
    assert "hobbies" in ENUM_MAPPINGS

    # 일부 대표적인 매핑 값 검증
    assert ENUM_MAPPINGS["gender"]["MALE"] == "남자"
    assert ENUM_MAPPINGS["gender"]["FEMALE"] == "여자"
    assert ENUM_MAPPINGS["ageGroup"]["AGE_20S"] == "20대"
    assert ENUM_MAPPINGS["personality"]["PASSIONATE"] == "열정적인"
