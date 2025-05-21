import os
import sys

# 프로젝트 루트 경로 추가 (core import 가능하게 함)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.vector_database import get_user_collection  # noqa: E402


def get_metadata(user_id: str):
    collection = get_user_collection()
    result = collection.get(ids=[user_id], include=["metadatas"])
    metadata = result["metadatas"][0] if result["metadatas"] else {}
    metadata = {k: v for k, v in metadata.items() if k not in ("field_embeddings")}

    if not metadata:
        print(f"❌ ID '{user_id}'에 대한 메타데이터를 찾을 수 없습니다.")
        return

    print(f"✅ Metadata for ID '{user_id}':")
    print(metadata)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("❗ 사용법: python get_metadata_by_id.py <user_id>")
        sys.exit(1)

    user_id = sys.argv[1]
    get_metadata(user_id)
