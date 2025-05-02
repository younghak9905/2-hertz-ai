# tests/test_enum_converter.py

from core.enum_process import convert_to_korean


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
    assert result == expected


def test_convert_list_fields():
    """리스트 필드 변환 테스트"""
    test_data = {"personality": ["CUTE", "RELIABLE"], "hobbies": ["GAMING", "MUSIC"]}

    expected = {"personality": ["아담한", "듬직한"], "hobbies": ["게임", "음악"]}

    result = convert_to_korean(test_data)
    assert result == expected


def test_handle_unknown_values():
    """알 수 없는 값 처리 테스트"""
    test_data = {"gender": "UNKNOWN", "personality": ["UNKNOWN", "RELIABLE"]}

    expected = {"gender": "UNKNOWN", "personality": ["UNKNOWN", "듬직한"]}

    result = convert_to_korean(test_data)
    assert result == expected
