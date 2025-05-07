# 임베딩 모델을 통해 유저 관심사를 임베딩 벡터화
from typing import List

from app.utils import logger


# 텍스트 생성 함수 (임베딩용 필드만 포함)
def convert_user_to_text(data: dict, fields: List[str]) -> str:
    """
    유저 정보 중 지정된 필드를 기반으로 텍스트 생성
    각 항목은 "field: value" 형식의 라인으로 구성됨
    """
    lines = []
    for key in fields:
        val = data.get(key, [])
        if isinstance(val, list):
            val = ", ".join(val)
        lines.append(f"{key}: {val}")
    return "\n".join(lines)


# 필드별 임베딩
@logger.log_performance(operation_name="embed_fields", include_memory=True)
def embed_fields(user: dict, fields: list, model=None) -> dict:
    """
    개별 필드를 임베딩 벡터로 변환

    Args:
        user: 사용자 정보 딕셔너리
        fields: 임베딩 처리 대상 필드 리스트
        model: SBERT 또는 호환 임베딩 모델

    Returns:
        field_embeddings: {필드명: 임베딩 벡터}
    """

    field_embeddings = {}
    # 모델 임베딩 차원 가져오기
    try:
        dim = model.get_sentence_embedding_dimension()
    except AttributeError:
        dim = 768  # fallback

    for field in fields:
        value = user.get(field)
        if not value:
            field_embeddings[field] = [0.0] * dim
            continue

        text = ", ".join(value) if isinstance(value, list) else str(value)

        try:
            embedding = model.encode(text).tolist()
            if len(embedding) != dim:
                raise ValueError(
                    f"Invalid dimension for field '{field}': {len(embedding)}"
                )
        except Exception as e:
            print(f"[embed_fields ERROR] Field: {field} / Error: {str(e)}")
            embedding = [0.0] * dim

        field_embeddings[field] = embedding

    return field_embeddings


# 테스트 코드
def test_embedding_output_shape():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    user = {"gender": "female", "currentInterests": ["reading", "hiking"]}
    fields = ["gender", "currentInterests"]
    embeddings = embed_fields(user, fields, model=model)

    for field, vec in embeddings.items():
        assert isinstance(vec, list)
        assert len(vec) == model.get_sentence_embedding_dimension()
