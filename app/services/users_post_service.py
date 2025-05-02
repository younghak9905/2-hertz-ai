# 사용자 등록 서비스 (임베딩 생성 및 유사도 계산)
import json
import time

import numpy as np
from fastapi import HTTPException

from app.core.embedding import convert_user_to_text, embed_fields
from app.core.enum_process import convert_to_korean
from app.core.matching_score import compute_matching_score
from app.core.vector_database import similarity_collection, user_collection
from app.models.sbert_loader import model


def safe_join(value):
    if isinstance(value, np.ndarray):
        return ", ".join(str(v) for v in value.tolist())  # ✅ ndarray → list
    elif isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def update_similarity_for_users(user_id: str):
    user_id = str(user_id)

    try:
        all_users = user_collection.get(include=["embeddings", "metadatas"])
        ids = all_users["ids"]
        embeddings = all_users["embeddings"]
        metadatas = all_users["metadatas"]

        if user_id not in ids:
            raise HTTPException(status_code=404, detail=f"User ID {user_id} not found")

        idx = ids.index(user_id)
        user_embedding = embeddings[idx]
        user_meta = metadatas[idx]

        # 1. 유사도 계산
        similarities = compute_matching_score(
            user_id=user_id,
            user_embedding=user_embedding,
            user_meta=user_meta,
            all_users=all_users,
        )

        # 2. 현재 유저 유사도 저장
        similarity_collection.upsert(
            ids=[user_id],
            embeddings=[user_embedding],
            metadatas=[{"userId": user_id, "similarities": json.dumps(similarities)}],
        )

        # 3. 역방향 유사도 삽입 또는 갱신
        for other_id, score in similarities.items():
            other_id = str(other_id)
            try:
                other_sim = similarity_collection.get(
                    ids=[other_id], include=["metadatas", "embeddings"]
                )

                # 새로 생성
                if not other_sim or not other_sim.get("metadatas"):
                    other_user = user_collection.get(
                        ids=[other_id], include=["embeddings", "metadatas"]
                    )

                    other_embedding = other_user["embeddings"][0]
                    similarity_collection.upsert(
                        ids=[other_id],
                        embeddings=[other_embedding],
                        metadatas=[
                            {
                                "userId": other_id,
                                "similarities": json.dumps({user_id: score}),
                            }
                        ],
                    )
                else:
                    other_meta = other_sim["metadatas"][0]
                    other_embedding = other_sim["embeddings"][0]
                    other_map = json.loads(other_meta.get("similarities", "{}"))
                    other_map[user_id] = score  # 역방향 삽입 또는 업데이트
                    similarity_collection.upsert(
                        ids=[other_id],
                        embeddings=[other_embedding],
                        metadatas=[
                            {
                                "userId": other_id,
                                "similarities": json.dumps(other_map),
                            }
                        ],
                    )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to update reverse similarity for {other_id}: {e}"
                )

        # 4. 다른 유저가 user_id에 대해 갖고 있던 유사도도 역으로 추가
        updated_map = dict(similarities)
        for other_id in ids:
            if str(other_id) == user_id:
                continue

            other_sim = similarity_collection.get(ids=[other_id])
            if not other_sim or not other_sim.get("metadatas"):
                continue

            other_meta = other_sim["metadatas"][0]
            other_map = json.loads(other_meta.get("similarities", "{}"))

            if user_id in other_map and other_id not in updated_map:
                updated_map[other_id] = other_map[user_id]

        # 5. 최종 반영 (역방향까지 보강한 최종 맵 저장)
        similarity_collection.upsert(
            ids=[user_id],
            embeddings=[user_embedding],
            metadatas=[{"userId": user_id, "similarities": json.dumps(updated_map)}],
        )

        return {"userId": user_id, "updated_similarities": len(updated_map)}

    except Exception as e:
        # 반드시 롤백 또는 오류 로깅
        print(
            f"❌ [SIMILARITY_UPDATE_ERROR] Failed to update similarity for {user_id}: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail={"code": "SIMILARITY_UPDATE_FAILED", "message": str(e)},
        )


async def register_user(user: dict) -> dict:
    start_time = time.time()
    user_id = str(user.get("userId"))

    # 1. 중복 ID 확인
    try:
        existing = user_collection.get(ids=[user_id])
        if existing and user_id in existing.get("ids", []):
            raise HTTPException(
                status_code=409,
                detail={"code": "EMBEDDING_CONFLICT_DUPLICATE_ID", "data": ""},
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"code": "USER_CHECK_FAILED", "message": str(e)},
        )

    try:
        # 2. 사용자 데이터 전처리 및 변환
        user = convert_to_korean(user)

        # 3. 임베딩 처리 대상 필드
        fields = [
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

        # 4. 텍스트로 변환 후 임베딩 생성
        user_text = convert_user_to_text(user, fields)
        embedding = model.encode(user_text).tolist()

        # 5. 개별 필드 임베딩
        field_embeddings = embed_fields(user, fields, model=model)

        # 6. 메타데이터 구성
        metadata = {k: safe_join(v) for k, v in user.items()}
        metadata["field_embeddings"] = json.dumps(field_embeddings)

        # 7. 사용자 DB 저장
        user_collection.add(
            ids=[user_id],
            embeddings=[embedding],
            metadatas=[metadata],
        )

    except Exception as e:
        print(f"[❌ REGISTER ERROR] 사용자 등록 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail={"code": "EMBEDDING_REGISTER_INTERNAL_ERROR", "message": str(e)},
        )

    try:
        # 8. 유사도 계산 및 저장 (양방향 포함)
        similarity_result = update_similarity_for_users(user_id)
    except Exception as e:
        print(f"[❌ SIMILARITY ERROR] 유사도 처리 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail={"code": "SIMILARITY_UPDATE_FAILED", "message": str(e)},
        )

    elapsed = round(time.time() - start_time, 3)

    return {
        "status": "registered",
        "userId": user_id,
        "matchedUserCount": similarity_result.get("updated_similarities", 0),
        "time_taken_seconds": elapsed,
    }
