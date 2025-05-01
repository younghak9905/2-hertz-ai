# 임베딩 모델을 통해 유저 관심사를 임베딩 벡터화


# 텍스트 생성 함수 (임베딩용 필드만 포함)
def convert_user_to_text(data: dict, fields: list) -> str:
    lines = []
    for key in fields:
        val = data.get(key, [])
        if isinstance(val, list):
            val = ", ".join(val)
        lines.append(f"{key}: {val}")
    return "\n".join(lines)


# 필드별 임베딩
def embed_fields(user: dict, fields: list, model=None) -> dict:
    field_embeddings = {}
    for field in fields:
        value = user.get(field)
        if not value:
            # 빈 값이면 768차원 0벡터 반환
            field_embeddings[field] = [0.0] * 768
            continue

        if isinstance(value, list):
            text = ", ".join(value)
        else:
            text = str(value)

        try:
            embedding = model.encode(text).tolist()
            if len(embedding) != 768:
                raise ValueError(
                    f"Unexpected embedding size for field '{field}': {len(embedding)}"
                )
        except Exception as e:
            print(f"[embed_fields error] Field: {field}, Error: {str(e)}")
            embedding = [0.0] * 768

        field_embeddings[field] = embedding

    return field_embeddings
