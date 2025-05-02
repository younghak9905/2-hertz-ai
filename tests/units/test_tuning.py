# tests/units/test_tuning.py
import json
from unittest.mock import MagicMock, patch

import pytest

from services.tuning_service import get_tuning_matches


class TestTuningService:
    @pytest.fixture
    def sample_similarities_data(self):
        """테스트용 유사도 데이터"""
        return {
            "metadatas": [
                {
                    "userId": "1",
                    "similarities": json.dumps(
                        {"2": 0.95, "3": 0.85, "4": 0.75, "5": 0.65, "6": 0.55}
                    ),
                }
            ]
        }

    @pytest.fixture
    def sample_users_data(self):
        """테스트용 사용자 데이터"""
        return {
            "metadatas": [
                {"userId": "1", "gender": "남자", "MBTI": "ESTP", "ageGroup": "20대"},
                {"userId": "2", "gender": "여자", "MBTI": "INFJ", "ageGroup": "20대"},
                {"userId": "3", "gender": "남자", "MBTI": "ENFP", "ageGroup": "30대"},
                {"userId": "4", "gender": "여자", "MBTI": "ISTP", "ageGroup": "20대"},
                {"userId": "5", "gender": "남자", "MBTI": "ESTP", "ageGroup": "20대"},
                {"userId": "6", "gender": "여자", "MBTI": "INTJ", "ageGroup": "40대"},
            ]
        }

    @patch("services.tuning_service.similarity_collection")
    @patch("services.tuning_service.user_collection")
    async def test_get_tuning_matches_all(
        self,
        mock_user_collection,
        mock_similarity_collection,
        sample_similarities_data,
        sample_users_data,
    ):
        """모든 카테고리 매칭 추천 테스트"""
        # 목 설정
        mock_similarity_collection.get.return_value = sample_similarities_data
        mock_user_collection.get.return_value = sample_users_data

        # 함수 실행
        result = await get_tuning_matches(1, "all")

        # 결과 검증
        assert result["code"] == "TUNING_SUCCESS"
        assert "userIdList" in result["data"]
        assert len(result["data"]["userIdList"]) <= 100
        assert 2 in result["data"]["userIdList"]  # 가장 높은 유사도를 가진 사용자

    @patch("services.tuning_service.similarity_collection")
    @patch("services.tuning_service.user_collection")
    async def test_get_tuning_matches_opposite_gender(
        self,
        mock_user_collection,
        mock_similarity_collection,
        sample_similarities_data,
        sample_users_data,
    ):
        """이성 매칭 추천 테스트"""
        # 목 설정
        mock_similarity_collection.get.return_value = sample_similarities_data
        mock_user_collection.get.return_value = sample_users_data

        # 함수 실행
        result = await get_tuning_matches(1, "opposite")

        # 결과 검증
        assert result["code"] == "TUNING_SUCCESS"
        assert "userIdList" in result["data"]

        # 모든 추천 대상이 여성인지 확인
        for user_id in result["data"]["userIdList"]:
            user_index = next(
                (
                    i
                    for i, u in enumerate(sample_users_data["metadatas"])
                    if int(u["userId"]) == user_id
                ),
                None,
            )
            assert user_index is not None
            assert sample_users_data["metadatas"][user_index]["gender"] == "여자"

    @patch("services.tuning_service.similarity_collection")
    @patch("services.tuning_service.user_collection")
    async def test_get_tuning_matches_same_gender(
        self,
        mock_user_collection,
        mock_similarity_collection,
        sample_similarities_data,
        sample_users_data,
    ):
        """동성 매칭 추천 테스트"""
        # 목 설정
        mock_similarity_collection.get.return_value = sample_similarities_data
        mock_user_collection.get.return_value = sample_users_data

        # 함수 실행
        result = await get_tuning_matches(1, "same")

        # 결과 검증
        assert result["code"] == "TUNING_SUCCESS"
        assert "userIdList" in result["data"]

        # 모든 추천 대상이 남성인지 확인
        for user_id in result["data"]["userIdList"]:
            user_index = next(
                (
                    i
                    for i, u in enumerate(sample_users_data["metadatas"])
                    if int(u["userId"]) == user_id
                ),
                None,
            )
            assert user_index is not None
            assert sample_users_data["metadatas"][user_index]["gender"] == "남자"

    @patch("services.tuning_service.similarity_collection")
    async def test_get_tuning_matches_user_not_found(self, mock_similarity_collection):
        """존재하지 않는 사용자 ID로 매칭 추천 실패 케이스 테스트"""
        # 사용자 없음으로 설정
        mock_similarity_collection.get.return_value = {"metadatas": []}

        # 함수 실행
        result = await get_tuning_matches(999)

        # 결과 검증
        assert result["code"] == "TUNING_SUCCESS_BUT_NO_MATCH"
        assert result["data"] is None
