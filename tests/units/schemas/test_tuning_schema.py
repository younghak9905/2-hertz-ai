from typing import Dict, List, Optional

import pytest
from pydantic import ValidationError

from schemas.tuning_schema import TuningMatching, TuningResponse


class TestTuningMatching:
    def test_valid_tuning_matching(self):
        """유효한 데이터로 TuningMatching 모델 생성 테스트"""
        data = {"userId": 123}
        matching = TuningMatching(**data)
        assert matching.userId == 123

    def test_invalid_tuning_matching_missing_userid(self):
        """userId 필드가 없는 경우 유효성 검증 실패 테스트"""
        with pytest.raises(ValidationError):
            TuningMatching()

    def test_invalid_tuning_matching_wrong_type(self):
        """잘못된 타입으로 userId 필드 제공 시 실패 테스트"""
        with pytest.raises(ValidationError):
            TuningMatching(userId="not-an-integer")


class TestTuningResponse:
    def test_valid_tuning_response(self):
        """유효한 데이터로 TuningResponse 모델 생성 테스트"""
        data = {"code": "TUNING_SUCCESS", "data": {"userIdList": [30, 1, 5, 6, 99, 56]}}
        response = TuningResponse(**data)
        assert response.code == "TUNING_SUCCESS"
        assert "userIdList" in response.data
        assert response.data["userIdList"] == [30, 1, 5, 6, 99, 56]

    def test_null_data_response(self):
        """매칭 결과가 없는 경우의 응답 테스트"""
        data = {"code": "TUNING_SUCCESS_BUT_NO_MATCH", "data": None}
        response = TuningResponse(**data)
        assert response.code == "TUNING_SUCCESS_BUT_NO_MATCH"
        assert response.data is None

    def test_invalid_tuning_response_missing_fields(self):
        """필수 필드 누락 시 유효성 검증 실패 테스트"""
        # code 필드 누락 (필수 필드)
        with pytest.raises(ValidationError):
            TuningResponse(data={"userIdList": [1, 2, 3]})

        # 모든 필드 누락
        with pytest.raises(ValidationError):
            TuningResponse()

    def test_invalid_tuning_response_wrong_structure(self):
        """잘못된 데이터 구조 제공 시 실패 테스트"""
        # data가 Dict[str, List[int]] 타입이 아닌 경우
        with pytest.raises(ValidationError):
            TuningResponse(code="TUNING_SUCCESS", data={"userIdList": "not-a-list"})

        # data 내부의 userIdList가 정수 리스트가 아닌 경우
        with pytest.raises(ValidationError):
            TuningResponse(
                code="TUNING_SUCCESS", data={"userIdList": ["not", "integers"]}
            )

    def test_example_data(self):
        """스키마 예시 데이터 검증 테스트"""
        # 성공 케이스 예시
        example1 = {
            "code": "TUNING_SUCCESS",
            "data": {"userIdList": [30, 1, 5, 6, 99, 56]},
        }
        response1 = TuningResponse(**example1)
        assert response1.model_dump() == example1

        # 매칭 없음 케이스 예시
        example2 = {"code": "TUNING_SUCCESS_BUT_NO_MATCH", "data": None}
        response2 = TuningResponse(**example2)
        assert response2.model_dump() == example2


def test_schema_examples():
    """모든 스키마의 예시 데이터가 실제로 유효한지 테스트"""
    # TuningMatching 예시 테스트
    tuning_example = {"userId": 1}
    assert TuningMatching(**tuning_example)

    # TuningResponse 예시 테스트 - 매칭 있음
    response_example1 = {
        "code": "TUNING_SUCCESS",
        "data": {"userIdList": [30, 1, 5, 6, 99, 56]},
    }
    assert TuningResponse(**response_example1)

    # TuningResponse 예시 테스트 - 매칭 없음
    response_example2 = {"code": "TUNING_SUCCESS_BUT_NO_MATCH", "data": None}
    assert TuningResponse(**response_example2)
