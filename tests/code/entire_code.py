import json
import time

import chromadb
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from utils.enum_converter import convert_to_korean

app = FastAPI()

# 모델 로드
model = SentenceTransformer("jhgan/ko-sbert-nli")

# ChromaDB 설정
chroma_client = chromadb.PersistentClient(path="./chroma_db")
user_collection = chroma_client.get_or_create_collection("user_profiles")
similarity_collection = chroma_client.get_or_create_collection("user_similarities")


# 필드별 임베딩
def embed_fields(data: dict, fields: list[str]) -> dict:
    field_embeddings = {}
    for field in fields:
        value = data.get(field, "")
        if isinstance(value, list):
            value = ", ".join(value)
        text = f"{field}: {value}"
        embedding = model.encode(text).tolist()
        field_embeddings[field] = embedding
    return field_embeddings


# 텍스트 생성 함수 (임베딩용 필드만 포함)
def convert_user_to_text(data: dict) -> str:
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
    lines = []
    for key in fields:
        val = data.get(key, [])
        if isinstance(val, list):
            val = ", ".join(val)
        lines.append(f"{key}: {val}")
    return "\n".join(lines)


# 규칙 기반 유사도 함수들----------------


# 축 | 설명 | 예시 점수
# E/I | 외향 ↔ 내향 | 0.5 중요
# N/S | 직관 ↔ 감각 | 1.0 중요 (정보 수용 방식 차이)
# F/T | 감정 ↔ 사고 | 1.0 중요 (판단 차이)
# J/P | 판단 ↔ 인식 | 0.5 중요
def mbti_weighted_score(mbti1, mbti2):
    MBTI_WEIGHTS = [0.5, 1.0, 1.0, 0.5]  # E/I, N/S, F/T, J/P
    if not mbti1 or not mbti2 or len(mbti1) != 4 or len(mbti2) != 4:
        return 0.0
    score = 0.0
    for i in range(4):
        if mbti1[i] == mbti2[i]:
            score += MBTI_WEIGHTS[i]
    return score / sum(MBTI_WEIGHTS)


# 같은 나이대면 1 아니면 0
def age_group_match_score(a, b):
    return 1.0 if a == b else 0.0


# preferredPeople - personality 같을수록 높은 점수
def match_tags(list1, list2):
    if not list1 or not list2:
        return 0.0
    overlap = set(list1).intersection(set(list2))
    return len(overlap) / len(
        set(list1).union(set(list2))
    )  # len(overlap) / max(len(set(list1 + list2)), 1)


# MBTI, ageGroup, preferredPeople-personalirty 는 지정된 매칭 스코어 계산 방법 적용
def rule_based_similarity(user1: dict, user2: dict) -> float:
    score = 0
    weight_total = 0

    score += mbti_weighted_score(user1.get("MBTI"), user2.get("MBTI")) * 1.0
    weight_total += 1.0

    score += age_group_match_score(user1.get("ageGroup"), user2.get("ageGroup")) * 1.0
    weight_total += 1.0

    score += match_tags(user1.get("preferredPeople", []), user2.get("personality", []))
    score += match_tags(user2.get("preferredPeople", []), user1.get("personality", []))
    weight_total += 1.0

    return score / weight_total if weight_total > 0 else 0.0


# 규칙 기반 유사도 함수들----------------


# 사용자 등록
@app.post("/api/v1/users")
async def register_user(user: dict, request: Request):
    try:
        start_time = time.time()

        # 1. 사용자 데이터 수신 및 ID 추출
        user = await request.json()
        userId = str(user.get("userId"))

        # 2. ID 중복 체크
        result = user_collection.get(ids=[userId])
        if result and userId in result.get("ids", []):
            raise HTTPException(
                status_code=400, detail=f"User ID {userId} already exists"
            )

        # 3. Enum → 한국어 변환
        user = convert_to_korean(user)

        # 4. 텍스트 임베딩
        text = convert_user_to_text(user)
        embedding = model.encode(text).tolist()

        # 필드별 임베딩 추가
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

        field_embeddings = embed_fields(user, fields)

        # 6. 메타데이터 구성 (원본 사용)
        metadata = {
            k: ", ".join(v) if isinstance(v, list) else v for k, v in user.items()
        }
        metadata["field_embeddings"] = json.dumps(field_embeddings)

        # 7. 사용자 등록
        user_collection.add(ids=[userId], embeddings=[embedding], metadatas=[metadata])

        # 8. 모든 사용자와의 유사도 계산
        all_users = user_collection.get(include=["embeddings", "metadatas"])
        all_ids = all_users["ids"]
        all_embeddings = np.array(all_users["embeddings"])
        all_metas = all_users["metadatas"]

        sims = cosine_similarity([embedding], all_embeddings)[0]
        similarities = {}

        for i, other_id in enumerate(all_ids):
            if other_id == userId:
                continue
            cosine_sim = float(sims[i])
            rule_sim = rule_based_similarity(user, all_metas[i])
            final_sim = 0.7 * cosine_sim + 0.3 * rule_sim
            similarities[other_id] = final_sim
            update_similarity_for_user(other_id)

        # 9. 유사도 저장
        similarity_collection.upsert(
            ids=[userId],
            embeddings=[embedding],
            metadatas=[{"userId": userId, "similarities": json.dumps(similarities)}],
        )

        elapsed = round(time.time() - start_time, 3)
        return {"status": "registered", "userId": userId, "time_taken_seconds": elapsed}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 사용자 추천
@app.get("/api/v1/tuning/{userId}")
async def tuning(userId: str, category: str = "all"):
    result = similarity_collection.get(ids=[userId], include=["metadatas"])
    if not result.get("metadatas"):
        raise HTTPException(status_code=404, detail="User not found")

    # 현재 사용자 정보
    base_user = user_collection.get(ids=[userId], include=["metadatas"])["metadatas"][0]
    base_gender = base_user.get("gender", "")

    # 유사도 목록
    metadata = result["metadatas"][0]
    similarities = json.loads(metadata.get("similarities", "{}"))
    sorted_similar = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[
        0:100
    ]

    # 추천 대상 ID 리스트
    recommended_ids = [uid for uid, _ in sorted_similar]
    all_users = user_collection.get(ids=recommended_ids, include=["metadatas"])[
        "metadatas"
    ]
    id_to_meta = {meta["userId"]: meta for meta in all_users}

    recommendations = []
    for uid, sim in sorted_similar:
        user_meta = id_to_meta.get(uid, {})
        gender = user_meta.get("gender", "")

        if category == "opposite" and gender == base_gender:
            continue
        if category == "same" and gender != base_gender:
            continue

        recommendations.append(
            {
                "userId": uid,
                "gender": gender,
                "MBTI": user_meta.get("MBTI", ""),
                "ageGroup": user_meta.get("ageGroup", ""),
                "similarity": round(sim, 4),
            }
        )

    return {"userId": userId, "category": category, "recommendations": recommendations}


@app.post("/tuning/recalculate")
def recalculate_similarities_for_all_users():
    all_users = user_collection.get(include=["embeddings", "metadatas"])
    ids = all_users.get("ids", [])
    embeddings = all_users.get("embeddings", [])
    metadatas = all_users.get("metadatas", [])

    if not ids or not embeddings:
        raise Exception("No user data available")

    embeddings_np = np.array(embeddings)

    for idx, userId in enumerate(ids):
        sims = cosine_similarity([embeddings_np[idx]], embeddings_np)[0]
        similarities = {}
        for i in range(len(ids)):
            if ids[i] == userId:
                continue
            cosine_sim = float(sims[i])
            rule_sim = rule_based_similarity(metadatas[idx], metadatas[i])
            final_sim = 0.7 * cosine_sim + 0.3 * rule_sim
            similarities[ids[i]] = final_sim

        similarity_collection.upsert(
            ids=[userId],
            embeddings=[embeddings[idx]],
            metadatas=[{"userId": userId, "similarities": json.dumps(similarities)}],
        )

    return {"status": "recalculated", "user_count": len(ids)}


@app.post("/similarity/update/{userId}")
def update_similarity_for_user(userId: str):
    all_users = user_collection.get(include=["embeddings", "metadatas"])
    ids = all_users.get("ids", [])
    embeddings = all_users.get("embeddings", [])
    metadatas = all_users.get("metadatas", [])

    if userId not in ids:
        raise HTTPException(status_code=404, detail=f"User ID {userId} not found")

    idx = ids.index(userId)
    target_embedding = embeddings[idx]
    target_meta = metadatas[idx]

    similarities = {}
    for i, other_id in enumerate(ids):
        if other_id == userId:
            continue
        cosine_sim = float(cosine_similarity([target_embedding], [embeddings[i]])[0][0])
        rule_sim = rule_based_similarity(target_meta, metadatas[i])
        final_sim = 0.7 * cosine_sim + 0.3 * rule_sim
        similarities[other_id] = final_sim

    similarity_collection.upsert(
        ids=[userId],
        embeddings=[target_embedding],
        metadatas=[{"userId": userId, "similarities": json.dumps(similarities)}],
    )

    return {
        "status": "updated",
        "userId": userId,
        "updated_similarities": len(similarities),
    }


# 두 사용자의 필드 임베딩을 불러와서 유사도만 계산해서 반환
@app.get("/similarity/field")
async def field_similarity(user1: str, user2: str, field: str):
    users = user_collection.get(ids=[user1, user2], include=["metadatas"])["metadatas"]
    if len(users) != 2:
        raise HTTPException(status_code=404, detail="User(s) not found")

    meta1 = users[0]
    meta2 = users[1]

    emb1 = json.loads(meta1.get("field_embeddings", "{}")).get(field)
    emb2 = json.loads(meta2.get("field_embeddings", "{}")).get(field)

    if not emb1 or not emb2:
        raise HTTPException(
            status_code=400, detail=f"Field '{field}' missing in one or both users."
        )

    sim = float(cosine_similarity([emb1], [emb2])[0][0])
    return {"user1": user1, "user2": user2, "field": field, "similarity": round(sim, 4)}


# 사용자 리스트 내부 확인용
@app.get("/users")
async def list_users():
    result = user_collection.get()
    return {"ids": result.get("ids", []), "count": len(result.get("ids", []))}


# 유사도 점수 내부 확인용
@app.get("/tuning/raw/{userId}")
async def get_raw_similarity(userId: str):
    result = similarity_collection.get(ids=[userId], include=["metadatas"])
    return result


# @app.post("/similarity/force-update")
# def force_update():
#     return recalculate_similarities_for_all_users()
