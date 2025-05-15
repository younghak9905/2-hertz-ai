# 임베딩 모델을 통해 유저 관심사를 임베딩 벡터화
from typing import List

from app.models.sbert_loader import get_model
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


@logger.log_performance(operation_name="embed_fields_optimized", include_memory=True)
def embed_fields_optimized(user: dict, fields: list) -> dict:
    """
    최적화된 필드별 임베딩 벡터 생성
    - 배치 처리로 한 번에 모든 필드 임베딩
    - 캐시 활용으로 중복 계산 방지

    Args:
        user: 사용자 정보 딕셔너리
        fields: 임베딩 처리 대상 필드 리스트

    Returns:
        field_embeddings: {필드명: 임베딩 벡터}
    """
    # 모델 인스턴스 획득
    model = get_model()

    # 처리할 텍스트와 필드 매핑
    field_texts = []
    field_mapping = []

    # 임베딩 차원 (모델에서 가져오기)
    dim = model.get_sentence_embedding_dimension()

    # 각 필드별 텍스트 준비
    for field in fields:
        value = user.get(field)
        if not value:
            continue

        text = ", ".join(value) if isinstance(value, list) else str(value)
        field_texts.append(text)
        field_mapping.append(field)

    # 배치 처리로 한 번에 임베딩 생성
    if field_texts:
        embeddings = model.encode(field_texts, show_progress_bar=False)

        # 결과 매핑
        field_embeddings = {}
        for i, field in enumerate(field_mapping):
            field_embeddings[field] = embeddings[i].tolist()

        # 누락된 필드에 대해 빈 벡터 추가
        for field in fields:
            if field not in field_embeddings:
                field_embeddings[field] = [0.0] * dim

        return field_embeddings
    else:
        # 모든 필드가 비어 있는 경우
        return {field: [0.0] * dim for field in fields}
