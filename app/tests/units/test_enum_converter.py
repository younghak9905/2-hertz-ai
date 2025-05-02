# tests/test_enum_converter.py
import pytest
from core.enum_process import ENUM_MAPPINGS, convert_to_korean


def test_convert_single_fields():
    """단일 필드 변환 테스트"""
    test_data = {
        "userId": 1,
        "gender": "MALE",
        "ageGroup": "AGE_20S",
        "religion": "NON_RELIGIOUS",
        "smoking": "NO_SMOKING",
        "drinking": "SOMETIMES",
    }

    expected = {
        "userId": 1,
        "gender": "남자",
        "ageGroup": "20대",
        "religion": "무교",
        "smoking": "비흡연",
        "drinking": "가끔",
    }

    result = convert_to_korean(test_data)
    print("Result of test1: ", result)
    assert result == expected


def test_convert_list_fields():
    """리스트 필드 변환 테스트"""
    test_data = {"personality": ["CUTE", "RELIABLE"], "hobbies": ["GAMING", "MUSIC"]}

    expected = {"personality": ["아담한", "듬직한"], "hobbies": ["게임", "음악"]}

    result = convert_to_korean(test_data)
    print("Result of test2: ", result)
    assert result == expected


def test_handle_unknown_values():
    """알 수 없는 값 처리 테스트"""
    test_data = {"gender": "UNKNOWN", "personality": ["UNKNOWN", "RELIABLE"]}

    expected = {"gender": "UNKNOWN", "personality": ["UNKNOWN", "듬직한"]}

    result = convert_to_korean(test_data)
    print("Result of test3: ", result)
    assert result == expected


def test_convert_multiple_fields():
    """다양한 필드 조합 변환 테스트"""
    test_data = {
        "userId": 1,
        "gender": "MALE",
        "ageGroup": "AGE_20S",
        "religion": "NON_RELIGIOUS",
        "smoking": "NO_SMOKING",
        "drinking": "SOMETIMES",
        "personality": ["CUTE", "RELIABLE"],
        "hobbies": ["GAMING", "MUSIC"],
        "likedSports": ["TENNIS", "YOGA"],
    }

    result = convert_to_korean(test_data)
    print("Result of test4: ", result)

    # 단일 값 필드 검증
    assert result["gender"] == "남자"
    assert result["ageGroup"] == "20대"
    assert result["religion"] == "무교"
    assert result["smoking"] == "비흡연"
    assert result["drinking"] == "가끔"

    # 리스트 필드 검증
    assert "아담한" in result["personality"]
    assert "듬직한" in result["personality"]
    assert "게임" in result["hobbies"]
    assert "음악" in result["hobbies"]
    assert "테니스" in result["likedSports"]
    assert "요가" in result["likedSports"]


def test_convert_partial_fields():
    """일부 필드만 포함된 경우 변환 테스트"""
    test_data = {
        "userId": 1,
        "gender": "MALE",
        "preferredPeople": ["NICE_VOICE", "PASSIONATE"],
    }

    result = convert_to_korean(test_data)
    print("Result of test5: ", result)

    assert result["gender"] == "남자"
    assert "목소리 좋은" in result["preferredPeople"]
    assert "열정적인" in result["preferredPeople"]


def test_enum_mapping_coverage():
    """Enum 매핑 테이블 검증"""
    # 필수 필드 매핑 검증
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

    # 일부 매핑 값 검증
    assert ENUM_MAPPINGS["gender"]["MALE"] == "남자"
    assert ENUM_MAPPINGS["gender"]["FEMALE"] == "여자"
    assert ENUM_MAPPINGS["ageGroup"]["AGE_20S"] == "20대"
    assert ENUM_MAPPINGS["personality"]["PASSIONATE"] == "열정적인"
