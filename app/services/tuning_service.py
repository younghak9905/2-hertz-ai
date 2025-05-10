# 매칭 서비스(유사도 기반 사용차 추천 로직)

import json

from fastapi import HTTPException

from app.core.vector_database import get_user_similarities, get_users_data
from app.utils import logger


# 유사도 정렬된 목록 가져오기
async def get_sorted_similar_users(
    user_id: str, top_k: int = 100
) -> list[tuple[str, float]]:
    sim_data = await get_user_similarities(user_id)

    if not sim_data or not sim_data.get("metadatas"):
        raise HTTPException(
            status_code=404,
            detail={
                "code": "TUNING_NOT_FOUND_USER",
                "data": None,
            },
        )

    try:
        sim_map = json.loads(sim_data["metadatas"][0].get("similarities", "{}"))
        sim_map = {str(k): float(v) for k, v in sim_map.items()}
    except (ValueError, TypeError, json.JSONDecodeError):
        raise HTTPException(status_code=500, detail="Invalid similarity data format")

    return sorted(sim_map.items(), key=lambda x: x[1], reverse=True)[:top_k]


async def get_users_metadata_map(user_ids: list[str]) -> dict[str, dict]:
    """
    사용자 ID 리스트를 받아 해당 사용자들의 메타데이터를 userId 기준으로 매핑한 딕셔너리를 반환

    Args:
        user_ids: 사용자 ID 리스트

    Returns:
        Dict[str, dict]: userId -> metadata 딕셔너리

    Raises:
        HTTPException: 메타데이터 조회 실패 시 500 에러 반환
    """
    if not user_ids:
        return {}

    try:
        user_data = await get_users_data(user_ids)
        metadata_list = user_data.get("metadatas", [])
        if not isinstance(metadata_list, list):
            raise ValueError("Invalid metadata format")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "USER_METADATA_FETCH_FAILED",
                "message": str(e),
            },
        )

    return {
        meta.get("userId"): meta
        for meta in metadata_list
        if isinstance(meta, dict) and "userId" in meta
    }


# 필터링 및 추천 포맷팅
def filter_and_format_recommendations(
    sorted_similar: list[tuple[str, float]],
    id_to_meta: dict,
) -> list[dict]:
    """
    유사도 기반 추천 목록에서 메타데이터가 존재하는 사용자만 필터링하고,
    추천 결과를 포맷팅하여 반환

    Args:
        sorted_similar: (user_id, similarity_score) 리스트
        id_to_meta: user_id -> metadata 매핑 딕셔너리

    Returns:
        추천 사용자 목록 (userId, similarity 포함된 딕셔너리)

    """

    user_id = []
    for uid, sim in sorted_similar:
        user_meta = id_to_meta.get(uid)
        if not user_meta:
            continue
        user_id.append(int(uid))

    return user_id


# 튜닝 추천 리스트 반환
@logger.log_performance(operation_name="get_matching_users", include_memory=True)
async def get_matching_users(user_id: str) -> list[dict]:
    similar_users = await get_sorted_similar_users(str(user_id), top_k=100)
    user_metadata_map = await get_users_metadata_map([uid for uid, _ in similar_users])

    return filter_and_format_recommendations(similar_users, user_metadata_map)
