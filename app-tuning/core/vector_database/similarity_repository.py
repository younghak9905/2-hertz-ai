import json

from fastapi import HTTPException

from .collections import get_similarity_collection


def clean_up_similarity(user_id: int) -> int:
    """
    다른 사용자들의 similarity 메타데이터에서 해당 user_id를 제거합니다.

    Args:
        user_id (int): 삭제 대상 사용자 ID

    Returns:
        int: 업데이트된 문서 수
    """
    user_id_str = str(user_id)
    updates_made = 0

    try:
        collection = get_similarity_collection()
        all_docs = collection.get(include=["metadatas"])
        ids = all_docs.get("ids", [])
        metadatas = all_docs.get("metadatas", [])

        if not ids:
            print("⚠️ similarity_collection에 문서가 없습니다.")
            return 0

        for doc_id, metadata in zip(ids, metadatas):
            similarities_json = metadata.get("similarities")
            if not similarities_json:
                continue

            try:
                similarities = json.loads(similarities_json)
            except json.JSONDecodeError:
                print(f"⚠️ ID '{doc_id}' - similarities JSON 파싱 실패")
                continue

            if user_id_str not in similarities:
                continue

            # user_id 제거 후 업데이트
            similarities.pop(user_id_str)
            metadata["similarities"] = json.dumps(similarities)

            try:
                collection.update(ids=[doc_id], metadatas=[metadata])
                updates_made += 1
            except Exception as update_err:
                print(f"❌ ID '{doc_id}' 업데이트 실패: {update_err}")

        print(f"✅ 총 {updates_made}개의 유사도 정보에서 user_id '{user_id}' 제거 완료")
        return updates_made

    except Exception as e:
        print(f"❌ similarity_collection 처리 중 오류 발생: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "EMBEDDING_CLEANUP_ERROR",
                "message": f"유사도 정리 중 오류 발생: {str(e)}",
            },
        )


async def get_user_similarities(user_id: str):
    """
    특정 사용자 ID에 대한 유사도 메타데이터 조회
    """
    collection = get_similarity_collection()
    return collection.get(ids=user_id, include=["metadatas"])


async def list_similarities():
    """
    전체 유사도 목록 조회
    """
    collection = get_similarity_collection()
    return collection.get()
