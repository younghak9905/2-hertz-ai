# 매칭 서비스(유사도 기반 사용차 추천 로직)

import json

from fastapi import HTTPException

from app.core.vector_database import get_user_similarities, get_users_data


# 유사도 정렬된 목록 가져오기
async def get_sorted_similar_users(
    user_id: str, top_k: int = 100
) -> list[tuple[str, float]]:
    sim_data = await get_user_similarities(user_id)

    if not sim_data or not sim_data.get("metadatas"):
        raise HTTPException(status_code=404, detail="Similarities not found")

    try:
        sim_map = json.loads(sim_data["metadatas"][0].get("similarities", "{}"))
        sim_map = {str(k): float(v) for k, v in sim_map.items()}  # Ensure proper types
    except (ValueError, TypeError, json.JSONDecodeError):
        raise HTTPException(status_code=500, detail="Invalid similarity data format")

    return sorted(sim_map.items(), key=lambda x: x[1], reverse=True)[:top_k]


async def get_users_metadata_map(user_ids: list[str]) -> dict:
    if not user_ids:
        return {}

    try:
        user_data = await get_users_data(user_ids)
        metadata_list = user_data.get("metadatas", [])
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to fetch user metadata")

    return {meta.get("userId"): meta for meta in metadata_list if "userId" in meta}


# 필터링 및 추천 포맷팅
def filter_and_format_recommendations(
    sorted_similar: list[tuple[str, float]],
    id_to_meta: dict,
) -> list[dict]:
    user_id = []

    for uid, sim in sorted_similar:
        user_meta = id_to_meta.get(uid, {})
        if not user_meta:
            continue

        user_id.append(int(uid))

    return user_id


# 튜닝 추천 리스트 반환
async def get_matching_users(user_id: str) -> list[dict]:
    similar_users = await get_sorted_similar_users(str(user_id), top_k=100)
    user_metadata_map = await get_users_metadata_map([uid for uid, _ in similar_users])

    return filter_and_format_recommendations(similar_users, user_metadata_map)
