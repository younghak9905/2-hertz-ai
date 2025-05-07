# users_post_service.py

import json
import time

import numpy as np
from fastapi import HTTPException

from app.core.embedding import convert_user_to_text, embed_fields
from app.core.enum_process import convert_to_korean
from app.core.matching_score import compute_matching_score
from app.core.vector_database import similarity_collection, user_collection
from app.models.sbert_loader import model
from app.schemas.user_schema import EmbeddingRegister


# 메타데이터 저장 시 문자열로 반환하기 위함
def safe_join(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return ", ".join(str(v) for v in value) if isinstance(value, list) else str(value)


# 매칭 스코어 정보 DB 저장
def upsert_similarity(user_id: str, embedding: list, similarities: dict):
    similarity_collection.upsert(
        ids=[user_id],
        embeddings=[embedding],
        metadatas=[{"userId": user_id, "similarities": json.dumps(similarities)}],
    )


# 매칭 스코어 정보 역방향 DB 저장
def update_reverse_similarities(user_id: str, similarities: dict):
    for other_id, score in similarities.items():
        try:
            other_id = str(other_id)
            other_sim = similarity_collection.get(
                ids=[other_id], include=["metadatas", "embeddings"]
            )

            if not other_sim or not other_sim.get("metadatas"):
                # 역 유저가 similarity DB에 없을 경우, user_collection에서 로드
                other_user = user_collection.get(ids=[other_id], include=["embeddings"])
                other_embedding = other_user["embeddings"][0]
                reverse_map = {user_id: score}
            else:
                other_meta = other_sim["metadatas"][0]
                other_embedding = other_sim["embeddings"][0]
                reverse_map = json.loads(other_meta.get("similarities", "{}"))
                reverse_map[user_id] = score

            upsert_similarity(other_id, other_embedding, reverse_map)

        except Exception as e:
            print(f"[REVERSE_SIMILARITY_UPDATE_ERROR]: {other_id} / {e}")
            raise HTTPException(
                status_code=500,
                detail={"code": "EMBEDDING_REGISTER_SERVER_ERROR", "message": str(e)},
            )


# 현재 유저가 저장하지 않은 상대방의 기존 유사도를 병합
def enrich_with_reverse_similarities(
    user_id: str, similarities: dict, all_users: dict
) -> dict:
    updated_map = dict(similarities)

    for other_id in all_users["ids"]:
        if str(other_id) == user_id:
            continue

        other_sim = similarity_collection.get(ids=[other_id])
        if not other_sim or not other_sim.get("metadatas"):
            continue

        other_meta = other_sim["metadatas"][0]
        other_map = json.loads(other_meta.get("similarities", "{}"))

        if user_id in other_map and other_id not in updated_map:
            updated_map[other_id] = other_map[user_id]

    return updated_map


# 전체 유저와의 매칭 스코어 계산 및 저장
def update_similarity_for_users(user_id: str) -> dict:
    try:
        all_users = user_collection.get(include=["embeddings", "metadatas"])
        ids, embeddings, metadatas = (
            all_users["ids"],
            all_users["embeddings"],
            all_users["metadatas"],
        )

        if user_id not in ids:
            raise HTTPException(status_code=404, detail=f"User ID {user_id} not found")

        idx = ids.index(user_id)
        user_embedding, user_meta = embeddings[idx], metadatas[idx]

        similarities = compute_matching_score(
            user_id=user_id,
            user_embedding=user_embedding,
            user_meta=user_meta,
            all_users=all_users,
        )

        # 현재 유저 유사도 저장
        upsert_similarity(user_id, user_embedding, similarities)

        # 역방향 저장
        update_reverse_similarities(user_id, similarities)

        # 반대방향에도 user_id가 존재하는 경우 통합
        updated_map = enrich_with_reverse_similarities(user_id, similarities, all_users)

        # 최종 반영
        upsert_similarity(user_id, user_embedding, updated_map)

        return {"userId": user_id, "updated_similarities": len(updated_map)}

    except Exception as e:
        print(f"[SIMILARITY_UPDATE_ERROR] {e}")
        raise HTTPException(
            status_code=500,
            detail={"code": "SIMILARITY_UPDATE_FAILED", "message": str(e)},
        )


# 신규 유저 등록과 매칭 스코어 계산 처리 통합 로직
async def register_user(user: EmbeddingRegister) -> dict:
    start_time = time.time()
    user_id = str(user.userId)

    # 중복 ID 체크
    existing = user_collection.get(ids=[user_id])
    if existing and user_id in existing.get("ids", []):
        raise HTTPException(
            status_code=409,
            detail={"code": "EMBEDDING_CONFLICT_DUPLICATE_ID", "data": None},
        )

    try:
        # Pydantic 모델 → dict 변환
        user_dict = user.model_dump()
        user_dict = convert_to_korean(user_dict)

        target_fields = [
            "emailDomain",
            "gender",
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

        user_text = convert_user_to_text(user_dict, target_fields)
        embedding = model.encode(user_text).tolist()
        field_embeddings = embed_fields(user_dict, target_fields, model=model)

        metadata = {k: safe_join(v) for k, v in user_dict.items()}
        metadata["field_embeddings"] = json.dumps(field_embeddings)

        user_collection.add(ids=[user_id], embeddings=[embedding], metadatas=[metadata])

    except Exception as e:
        print(f"[ REGISTER ERROR] 사용자 등록 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail={"code": "EMBEDDING_REGISTER_SERVER_ERROR", "message": str(e)},
        )

    try:
        similarity_result = update_similarity_for_users(user_id)
    except Exception as e:
        print(f"[ SIMILARITY ERROR] 유사도 처리 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "EMBEDDING_REGISTER_SIMILARITY_UPDATE_FAILED",
                "message": str(e),
            },
        )

    elapsed = round(time.time() - start_time, 3)

    return {
        "status": "registered",
        "userId": user_id,
        "matchedUserCount": similarity_result.get("updated_similarities", 0),
        "time_taken_seconds": elapsed,
    }
