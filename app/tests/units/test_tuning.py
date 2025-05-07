# tests/units/test_tuning_service.py
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.tuning_service import (
    filter_and_format_recommendations,
    get_matching_users,
    get_sorted_similar_users,
    get_users_metadata_map,
)


@pytest.fixture
def similar_users_data():
    """유사도 정보 테스트 데이터"""
    return {
        "ids": ["1"],
        "metadatas": [
            {
                "userId": "1",
                "similarities": json.dumps(
                    {"2": 0.95, "3": 0.85, "4": 0.75, "5": 0.65, "6": 0.55}
                ),
            }
        ],
    }


@pytest.fixture
def users_metadata():
    """사용자 메타데이터 테스트 데이터"""
    return {
        "ids": ["2", "3", "4", "5", "6"],
        "metadatas": [
            {"userId": "2", "gender": "여자", "MBTI": "INFJ", "ageGroup": "20대"},
            {"userId": "3", "gender": "남자", "MBTI": "ENFP", "ageGroup": "30대"},
            {"userId": "4", "gender": "여자", "MBTI": "ISTP", "ageGroup": "20대"},
            {"userId": "5", "gender": "남자", "MBTI": "ESTP", "ageGroup": "20대"},
            {"userId": "6", "gender": "여자", "MBTI": "INTJ", "ageGroup": "40대"},
        ],
    }


class TestTuningService:
    @pytest.mark.asyncio
    @patch("app.services.tuning_service.get_user_similarities")
    async def test_get_sorted_similar_users(
        self, mock_get_user_similarities, similar_users_data
    ):
        """유사도 정렬 목록 가져오기 테스트"""
        # 목 설정
        mock_get_user_similarities.return_value = similar_users_data

        # 함수 실행
        result = await get_sorted_similar_users("1", top_k=3)

        # 결과 검증
        assert len(result) == 3
        assert result[0][0] == "2"  # 첫 번째로 유사한 사용자
        assert result[0][1] == 0.95  # 유사도 점수
        assert result[1][0] == "3"
        assert result[2][0] == "4"

    @pytest.mark.asyncio
    @patch("app.services.tuning_service.get_users_data")
    async def test_get_users_metadata_map(self, mock_get_users_data, users_metadata):
        """사용자 메타데이터 맵 가져오기 테스트"""
        # 목 설정
        mock_get_users_data.return_value = users_metadata

        # 함수 실행
        result = await get_users_metadata_map(["2", "3", "4"])

        # 결과 검증
        assert len(result) == 5  # 모든 메타데이터가 반환됨
        assert "2" in result
        assert result["2"]["gender"] == "여자"
        assert result["3"]["MBTI"] == "ENFP"

    def test_filter_and_format_recommendations(self, users_metadata):
        """추천 사용자 필터링 및 포맷팅 테스트"""
        # 테스트 데이터
        sorted_similar = [("2", 0.95), ("3", 0.85), ("4", 0.75)]

        # 메타데이터 맵 생성
        id_to_meta = {meta["userId"]: meta for meta in users_metadata["metadatas"]}

        # 함수 실행
        result = filter_and_format_recommendations(sorted_similar, id_to_meta)

        # 결과 검증
        assert len(result) == 3
        assert 2 in result
        assert 3 in result
        assert 4 in result

    @pytest.mark.asyncio
    @patch("app.services.tuning_service.get_sorted_similar_users")
    @patch("app.services.tuning_service.get_users_metadata_map")
    async def test_get_matching_users(
        self, mock_get_users_metadata_map, mock_get_sorted_similar_users, users_metadata
    ):
        """매칭 사용자 목록 반환 테스트"""
        # 목 설정
        mock_get_sorted_similar_users.return_value = [
            ("2", 0.95),
            ("3", 0.85),
            ("4", 0.75),
        ]

        # 메타데이터 맵 생성
        id_to_meta = {meta["userId"]: meta for meta in users_metadata["metadatas"]}
        mock_get_users_metadata_map.return_value = id_to_meta

        # 함수 실행
        result = await get_matching_users("1")

        # 결과 검증
        assert len(result) == 3
        assert 2 in result
        assert 3 in result
        assert 4 in result
