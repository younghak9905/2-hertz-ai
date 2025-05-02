# tests/units/test_users_post.py
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from core.embedding import convert_user_to_text, embed_fields


class TestEmbedding:
    @pytest.fixture
    def sample_user_data(self):
        """테스트용 사용자 데이터"""
        return {
            "userId": 1,
            "emailDomain": "kakaotech.com",
            "gender": "남자",
            "ageGroup": "20대",
            "MBTI": "ESTP",
            "religion": "무교",
            "smoking": "비흡연",
            "drinking": "가끔",
            "personality": ["아담한", "듬직한"],
            "preferredPeople": ["목소리 좋은", "욕 안하는", "열정적인"],
            "currentInterests": ["베이킹", "그림그리기", "반려식물"],
            "favoriteFoods": ["과일", "양식", "길거리음식"],
            "likedSports": ["볼링", "당구", "요가"],
            "pets": ["물고기", "햄스터", "토끼"],
            "selfDevelopment": ["독서", "공부", "카공"],
            "hobbies": ["게임", "음악"],
        }

    def test_convert_user_to_text(self, sample_user_data):
        """사용자 데이터를 텍스트로 변환하는 함수 테스트"""
        fields = [
            "emailDomain",
            "gender",
            "MBTI",
            "religion",
            "smoking",
            "drinking",
            "currentInterests",
            "favoriteFoods",
            "likedSports",
            "pets",
            "selfDevelopment",
            "hobbies",
        ]
        text = convert_user_to_text(sample_user_data, fields)
        print("text: \n", text)

        # 결과 검증
        assert "gender: 남자" in text
        assert "MBTI: ESTP" in text
        assert "currentInterests: 베이킹, 그림그리기, 반려식물" in text
        assert len(text.split("\n")) == len(fields)  # 지정한 필드 수만큼 라인 생성

    @patch("sentence_transformers.SentenceTransformer")
    def test_embed_fields(self, mock_model, sample_user_data):
        """각 필드별 임베딩 생성 함수 테스트"""
        # 목 모델 설정
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.zeros(768)  # 768 차원 임베딩 벡터

        fields = ["gender", "MBTI", "currentInterests"]
        field_embeddings = embed_fields(sample_user_data, fields, model=mock_encoder)
        print("field_embedding: ", field_embeddings)

        # 결과 검증
        assert set(field_embeddings.keys()) == set(fields)
        assert len(field_embeddings["gender"]) == 768
        assert len(field_embeddings["MBTI"]) == 768
        assert len(field_embeddings["currentInterests"]) == 768

        # 모델 호출 검증
        assert mock_encoder.encode.call_count == 3

    @patch("sentence_transformers.SentenceTransformer")
    def test_embed_fields_with_empty_value(self, mock_model, sample_user_data):
        """빈 값이 있는 경우의 임베딩 생성 테스트"""
        # 빈 값이 있는 데이터
        sample_user_data["hobbies"] = []

        # 목 모델 설정
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.zeros(768)

        fields = ["gender", "hobbies"]
        field_embeddings = embed_fields(sample_user_data, fields, model=mock_encoder)
        print("field_embedding: ", field_embeddings)

        # 빈 값에 대해 0 벡터 반환 검증
        assert len(field_embeddings["hobbies"]) == 768
        assert all(v == 0 for v in field_embeddings["hobbies"])

        # 모델은 빈 값이 아닌 필드에 대해서만 호출
        assert mock_encoder.encode.call_count == 1
