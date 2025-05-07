"""
사용자 매칭 정보 튜닝 서비스 테스트 모듈
이 모듈은 유사도 기반 사용자 추천 및 매칭 관련 서비스를 단위 테스트합니다.
주요 기능:
- 유사도가 높은 사용자 정렬 및 필터링
- 사용자 메타데이터 조회 및 변환
- 추천 사용자 정보 포맷팅
- 최종 매칭 사용자 목록 생성
"""

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
    """
    유사도 정보 테스트 데이터 픽스처
    특정 사용자와 다른 사용자들 간의 유사도 점수 정보 제공
    유사도 점수는 0~1 사이의 값으로, 값이 클수록 더 유사함을 의미
    """
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
    """
    사용자 메타데이터 테스트 데이터 픽스처
    유사도가 계산된 사용자들의 프로필 정보 제공
    성별, MBTI, 연령대 등 매칭에 사용되는 기본 메타데이터 포함
    """
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
    """
    튜닝 서비스 테스트 클래스
    유사도 기반 사용자 추천 및 매칭 관련 기능을 검증하는 테스트 케이스 모음
    """

    @pytest.mark.asyncio
    @patch("app.services.tuning_service.get_user_similarities")
    async def test_get_sorted_similar_users(
        self, mock_get_user_similarities, similar_users_data
    ):
        """
        유사도 정렬 목록 가져오기 테스트
        유사도 점수가 높은 순으로 정렬된 사용자 목록 반환 기능 검증
        top_k 파라미터를 통한 결과 제한 기능 테스트
        """
        # 유저 유사도 조회 함수 모킹
        mock_get_user_similarities.return_value = similar_users_data

        # 테스트 대상 함수 실행 - 상위 3명만 요청
        result = await get_sorted_similar_users("1", top_k=3)

        # 결과 검증
        assert len(result) == 3  # 요청한 top_k(3)만큼 결과가 반환되는지 확인
        assert result[0][0] == "2"  # 첫 번째로 유사한 사용자
        assert result[0][1] == 0.95  # 첫 번째 사용자의 유사도 점수
        assert result[1][0] == "3"  # 두 번째로 유사한 사용자
        assert result[2][0] == "4"  # 세 번째로 유사한 사용자
        # 정렬 순서 검증 - 유사도 내림차순 정렬 확인

    @pytest.mark.asyncio
    @patch("app.services.tuning_service.get_users_data")
    async def test_get_users_metadata_map(self, mock_get_users_data, users_metadata):
        """
        사용자 메타데이터 맵 가져오기 테스트
        사용자 ID를 키로, 메타데이터를 값으로 하는 맵 생성 기능 검증
        여러 사용자 ID에 대한 메타데이터 일괄 조회 테스트
        """
        # 사용자 데이터 조회 함수 모킹
        mock_get_users_data.return_value = users_metadata

        # 테스트 대상 함수 실행 - 특정 ID 목록에 대한 메타데이터 요청
        result = await get_users_metadata_map(["2", "3", "4"])

        # 결과 검증
        assert len(result) == 5  # 전체 사용자 메타데이터 맵
        assert "2" in result  # 요청한 ID가 결과에 포함되는지 확인
        assert result["2"]["gender"] == "여자"  # 메타데이터 값 확인
        assert result["3"]["MBTI"] == "ENFP"  # 다른 필드 값 확인

    def test_filter_and_format_recommendations(self, users_metadata):
        """
        추천 사용자 필터링 및 포맷팅 테스트
        유사도 정보와 메타데이터를 결합하여 추천 결과 포맷 생성 검증
        사용자 ID를 정수형으로 변환하는 기능 테스트
        """
        # 테스트 데이터 - 유사도 정렬된 사용자 목록 (ID, 유사도 점수)
        sorted_similar = [("2", 0.95), ("3", 0.85), ("4", 0.75)]

        # 메타데이터 맵 생성 - ID를 키로 하는 딕셔너리
        id_to_meta = {meta["userId"]: meta for meta in users_metadata["metadatas"]}

        # 테스트 대상 함수 실행
        result = filter_and_format_recommendations(sorted_similar, id_to_meta)

        # 결과 검증
        assert len(result) == 3  # 입력한 사용자 수만큼 결과가 반환되는지 확인
        assert 2 in result  # ID가 정수형으로 변환되었는지 확인
        assert 3 in result
        assert 4 in result
        # 각 사용자 ID에 대한 추천 정보가 올바르게 포맷팅되었는지 확인

    @pytest.mark.asyncio
    @patch("app.services.tuning_service.get_sorted_similar_users")
    @patch("app.services.tuning_service.get_users_metadata_map")
    async def test_get_matching_users(
        self, mock_get_users_metadata_map, mock_get_sorted_similar_users, users_metadata
    ):
        """
        매칭 사용자 목록 반환 테스트
        유사도 정렬 및 메타데이터 결합을 통한 최종 매칭 사용자 목록 생성 검증
        전체 매칭 프로세스 통합 테스트
        """
        # 정렬된 유사 사용자 목록 조회 함수 모킹
        mock_get_sorted_similar_users.return_value = [
            ("2", 0.95),
            ("3", 0.85),
            ("4", 0.75),
        ]

        # 메타데이터 맵 생성 및 모킹
        id_to_meta = {meta["userId"]: meta for meta in users_metadata["metadatas"]}
        mock_get_users_metadata_map.return_value = id_to_meta

        # 테스트 대상 함수 실행 - 사용자 1에 대한 매칭 목록 요청
        result = await get_matching_users("1")

        # 결과 검증
        assert len(result) == 3  # 매칭된 사용자 수 확인
        assert 2 in result  # 매칭 결과에 포함된 사용자 ID 확인
        assert 3 in result
        assert 4 in result
        # 각 매칭 사용자에 대한 필요한 정보가 모두 포함되어 있는지 확인
