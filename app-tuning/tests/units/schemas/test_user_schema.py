import pytest
from pydantic import ValidationError
from schemas.user_schema import BaseResponse, EmbeddingRegister, ErrorResponse


class TestUserSchema:
    def test_valid_embedding_register(self):
        """유효한 사용자 등록 데이터에 대한 테스트"""
        valid_data = {
            "userId": 1,
            "emailDomain": "user1@kakaotech.com",
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

        # 예외가 발생하지 않아야 합니다
        user = EmbeddingRegister(**valid_data)

        # 값이 예상대로 저장되었는지 확인
        assert user.userId == 1
        assert user.emailDomain == "user1@kakaotech.com"
        assert "GAMING" in user.hobbies

    def test_invalid_embedding_register(self):
        """필수 필드가 누락된 경우 테스트"""
        invalid_data = {
            "userId": 1,
            # 다른 필수 필드 누락
        }

        # ValidationError가 발생해야 합니다
        with pytest.raises(ValidationError):
            EmbeddingRegister(**invalid_data)

    # BaseResponse 테스트 추가
    def test_base_response(self):
        """기본 응답 형식 테스트"""
        # 성공 응답 테스트
        success_data = {"code": "EMBEDDING_REGISTER_SUCCESS", "data": None}
        response = BaseResponse(**success_data)
        assert response.code == "EMBEDDING_REGISTER_SUCCESS"
        assert response.data is None

        # 데이터가 있는 응답 테스트
        data_response = {"code": "TUNING_SUCCESS", "data": {"userIdList": [1, 2, 3, 4]}}
        response = BaseResponse(**data_response)
        assert response.code == "TUNING_SUCCESS"
        assert response.data == {"userIdList": [1, 2, 3, 4]}

        # 유효하지 않은 응답 테스트 (code 필드 누락)
        with pytest.raises(ValidationError):
            BaseResponse(data={"userIdList": [1, 2, 3]})

    # ErrorResponse 테스트 추가
    def test_error_response(self):
        """오류 응답 형식 테스트"""
        # 기본 오류 응답 테스트
        error_data = {
            "code": "EMBEDDING_REGISTER_BAD_REQUEST",
            "data": {"message": "Invalid email domain"},
        }
        error_response = ErrorResponse(**error_data)
        assert error_response.code == "EMBEDDING_REGISTER_BAD_REQUEST"
        assert error_response.data["message"] == "Invalid email domain"

        # data가 없는 오류 응답 테스트
        error_without_data = {"code": "EMBEDDING_DELETE_NOT_FOUND", "data": None}
        error_response = ErrorResponse(**error_without_data)
        assert error_response.code == "EMBEDDING_DELETE_NOT_FOUND"
        assert error_response.data is None

        # 유효하지 않은 오류 응답 테스트 (code 필드 누락)
        with pytest.raises(ValidationError):
            ErrorResponse(data={"message": "Something went wrong"})
