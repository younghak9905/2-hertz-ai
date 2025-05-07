# tests/units/test_users_post_service.py
import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi import HTTPException

from app.services.users_post_service import (
    enrich_with_reverse_similarities,
    register_user,
    safe_join,
    update_reverse_similarities,
    update_similarity_for_users,
    upsert_similarity,
)


@pytest.fixture
def sample_user_data():
    """테스트용 사용자 등록 데이터"""
    return {
        "userId": 1,
        "emailDomain": "kakaotech.com",
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


@pytest.fixture
def embedding_fixture():
    """임베딩 테스트 데이터"""
    return [0.1] * 768  # 768차원 임베딩 벡터


@pytest.fixture
def similarity_collection_fixture():
    """유사도 컬렉션 목 객체"""
    mock = MagicMock()
    mock.get.return_value = {
        "ids": ["1", "2"],
        "metadatas": [
            {"userId": "1", "similarities": json.dumps({"2": 0.95, "3": 0.85})},
            {"userId": "2", "similarities": json.dumps({"1": 0.95, "4": 0.75})},
        ],
        "embeddings": [[0.1] * 768, [0.2] * 768],
    }
    mock.upsert = MagicMock()
    return mock


@pytest.fixture
def user_collection_fixture():
    """사용자 컬렉션 목 객체"""
    mock = MagicMock()
    mock.get.return_value = {
        "ids": ["1", "2", "3", "4"],
        "metadatas": [
            {
                "userId": "1",
                "emailDomain": "kakaotech.com",
                "MBTI": "ESTP",
                "field_embeddings": json.dumps(
                    {"interests": [0.1] * 768, "hobbies": [0.2] * 768}
                ),
            },
            {
                "userId": "2",
                "emailDomain": "kakaotech.com",
                "MBTI": "INFJ",
                "field_embeddings": json.dumps(
                    {"interests": [0.3] * 768, "hobbies": [0.4] * 768}
                ),
            },
            {
                "userId": "3",
                "emailDomain": "kakaotech.com",
                "MBTI": "ENFP",
                "field_embeddings": json.dumps(
                    {"interests": [0.5] * 768, "hobbies": [0.6] * 768}
                ),
            },
            {
                "userId": "4",
                "emailDomain": "other.com",  # 다른 도메인
                "MBTI": "ISTP",
                "field_embeddings": json.dumps(
                    {"interests": [0.7] * 768, "hobbies": [0.8] * 768}
                ),
            },
        ],
        "embeddings": [[0.1] * 768, [0.2] * 768, [0.3] * 768, [0.4] * 768],
    }
    mock.add = MagicMock()
    return mock


class TestUsersPostService:
    def test_safe_join(self):
        """메타데이터 저장 시 문자열로 반환 테스트"""
        # 리스트 입력
        assert safe_join(["a", "b", "c"]) == "a, b, c"

        # numpy 배열 입력
        np_array = np.array([1, 2, 3])
        assert safe_join(np_array) == "1, 2, 3"

        # 단일 값 입력
        assert safe_join("test") == "test"
        assert safe_join(123) == "123"

    @patch("app.services.users_post_service.similarity_collection")
    def test_upsert_similarity(self, mock_similarity_collection):
        """매칭 스코어 정보 DB 저장 테스트"""
        # 테스트 데이터
        user_id = "1"
        embedding = [0.1] * 768
        similarities = {"2": 0.95, "3": 0.85}

        # 함수 실행
        upsert_similarity(user_id, embedding, similarities)

        # 목 호출 검증
        mock_similarity_collection.upsert.assert_called_once()
        call_args = mock_similarity_collection.upsert.call_args[1]
        assert call_args["ids"] == [user_id]
        assert call_args["embeddings"] == [embedding]
        assert json.loads(call_args["metadatas"][0]["similarities"]) == similarities

    @patch("app.services.users_post_service.similarity_collection")
    def test_update_reverse_similarities(
        self, mock_similarity_collection, similarity_collection_fixture
    ):
        """매칭 스코어 정보 역방향 DB 저장 테스트"""
        # 목 설정
        mock_similarity_collection.get.return_value = {
            "metadatas": [{"similarities": json.dumps({"4": 0.7})}],
            "embeddings": [[0.2] * 768],
        }

        # 테스트 데이터
        user_id = "1"
        similarities = {"2": 0.95, "3": 0.85}

        # 함수 실행
        update_reverse_similarities(user_id, similarities)

        # 목 호출 검증
        assert mock_similarity_collection.get.call_count == len(similarities)
        assert mock_similarity_collection.upsert.call_count == len(similarities)

    def test_enrich_with_reverse_similarities(self, similarity_collection_fixture):
        """현재 유저가 저장하지 않은 상대방의 기존 유사도를 병합 테스트"""
        # 테스트 데이터
        user_id = "1"
        similarities = {"2": 0.95}
        all_users = {"ids": ["1", "2", "3", "4"]}

        # 목 설정
        with patch(
            "app.services.users_post_service.similarity_collection",
            similarity_collection_fixture,
        ):
            # 함수 실행
            result = enrich_with_reverse_similarities(user_id, similarities, all_users)

            # 결과 검증
            assert "2" in result
            assert result["2"] == 0.95
            assert "3" in result  # 역방향에서 추가된 항목

    @patch("app.services.users_post_service.compute_matching_score")
    @patch("app.services.users_post_service.user_collection")
    @patch("app.services.users_post_service.similarity_collection")
    def test_update_similarity_for_users(
        self,
        mock_similarity_collection,
        mock_user_collection,
        mock_compute_matching_score,
        user_collection_fixture,
    ):
        """전체 유저와의 매칭 스코어 계산 및 저장 테스트"""
        # 목 설정
        mock_user_collection.get.return_value = user_collection_fixture.get()
        mock_compute_matching_score.return_value = {"2": 0.95, "3": 0.85}

        # 함수 실행
        result = update_similarity_for_users("1")

        # 결과 검증
        assert result["userId"] == "1"
        assert "updated_similarities" in result
        assert mock_similarity_collection.upsert.call_count >= 1

    @pytest.mark.asyncio
    @patch("app.services.users_post_service.user_collection")
    @patch("app.services.users_post_service.update_similarity_for_users")
    @patch("app.services.users_post_service.model")
    @patch("app.services.users_post_service.convert_to_korean")
    @patch("app.services.users_post_service.convert_user_to_text")
    @patch("app.services.users_post_service.embed_fields")
    async def test_register_user(
        self,
        mock_embed_fields,
        mock_convert_user_to_text,
        mock_convert_to_korean,
        mock_model,
        mock_update_similarity,
        mock_user_collection,
        sample_user_data,
        embedding_fixture,
    ):
        """신규 유저 등록과 매칭 스코어 계산 처리 통합 로직 테스트"""
        # 목 설정
        mock_user_collection.get.return_value = {"ids": []}  # 신규 사용자 (중복 없음)
        mock_convert_to_korean.return_value = sample_user_data
        mock_convert_user_to_text.return_value = "사용자 텍스트"
        mock_model.encode.return_value = np.array(embedding_fixture)
        mock_embed_fields.return_value = {"interests": [0.1] * 768}
        mock_update_similarity.return_value = {
            "userId": "1",
            "updated_similarities": 3,
        }

        # Pydantic 모델 목 생성
        user_model = MagicMock()
        user_model.model_dump.return_value = sample_user_data
        user_model.userId = 1

        # 함수 실행
        result = await register_user(user_model)

        # 결과 검증
        assert result["status"] == "registered"
        assert result["userId"] == "1"
        assert result["matchedUserCount"] == 3
        assert "time_taken_seconds" in result
        mock_user_collection.add.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.services.users_post_service.user_collection")
    async def test_register_user_duplicate(
        self, mock_user_collection, sample_user_data
    ):
        """중복 사용자 등록 시 예외 발생 테스트"""
        # 목 설정
        mock_user_collection.get.return_value = {"ids": ["1"]}  # 이미 존재하는 사용자

        # Pydantic 모델 목 생성
        user_model = MagicMock()
        user_model.userId = 1

        # 예외 발생 검증
        with pytest.raises(HTTPException) as excinfo:
            await register_user(user_model)

        # 예외 상태 코드 및 메시지 검증
        assert excinfo.value.status_code == 409
        assert "CONFLICT" in excinfo.value.detail["code"]
