import os
import sys
import traceback

script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.insert(0, project_root)
from core.matching_score_by_category import compute_matching_score
from core.vector_database import get_user_collection
from fastapi import HTTPException
from services.user_service import upsert_similarity_v3
from utils.logger import log_performance, logger

# SIM_COLLECTIONS = {"friend": "friend_similarities", "couple": "couple_similarities"} # get_similarity_collection 등으로 대체


def get_all_user_ids():
    data = get_user_collection().get(include=[])
    return data["ids"]


def recompute_all_similarities(mode: str):
    """
    모든 유저 데이터를 한 번만 로드하여 유사도를 재계산합니다.
    """
    logger.info(f"✅ Recomputing {mode} similarities...")

    # 1. 성능 개선: 모든 유저 정보를 한 번만 가져옵니다.
    try:
        all_users = get_user_collection().get(include=["embeddings", "metadatas"])
    except Exception as e:
        logger.error(f"[ERROR] Failed to fetch all users: {e}")
        return

    user_ids = all_users["ids"]

    for user_id in user_ids:
        try:
            # 2. 개선: all_users 데이터를 파라미터로 전달합니다.
            result = update_similarity_for_single_user(
                user_id=user_id, category=mode, all_users_data=all_users
            )
            logger.info(
                f"[{mode.upper()}] Updated: {user_id} with {result['updated_similarities']} matches"
            )
        except Exception as e:
            logger.info(f"[ERROR] {mode} similarity failed for {user_id}: {e}")
            traceback.print_exc()


@log_performance(
    operation_name="update_similarity_for_single_user", include_memory=True
)
def update_similarity_for_single_user(
    user_id: str, category: str, all_users_data: dict
) -> dict:
    """
    단일 유저에 대해 다른 모든 유저와의 유사도를 계산하고 저장합니다. (단방향)
    """
    try:
        ids = all_users_data["ids"]
        if user_id not in ids:
            # 에러 처리는 기존과 같이 유지
            raise HTTPException(status_code=404, detail=...)

        idx = ids.index(user_id)
        user_embedding, user_meta = (
            all_users_data["embeddings"][idx],
            all_users_data["metadatas"][idx],
        )

        # 3. 로직 단순화: 정방향 계산만 수행합니다.
        similarities = compute_matching_score(
            user_id=user_id,
            user_embedding=user_embedding,
            user_meta=user_meta,
            all_users=all_users_data,
            category=category,  # <--- 이 라인을 추가하세요
        )

        # 4. 단순화 및 수정: `category`를 사용해 해당 컬렉션에 한 번만 저장합니다.
        upsert_similarity_v3(
            user_id=user_id,
            embedding=user_embedding,
            similarities=similarities,
            category=category,
        )

        return {"userId": user_id, "updated_similarities": len(similarities)}

    except Exception as e:
        # 에러 처리는 기존과 같이 유지
        logger.info(f"[SIMILARITY_UPDATE_ERROR] {e}")
        raise HTTPException(status_code=500, detail=...)


if __name__ == "__main__":
    recompute_all_similarities("friend")
    recompute_all_similarities("couple")
    logger.info("🎉 All similarity recomputations completed.")
