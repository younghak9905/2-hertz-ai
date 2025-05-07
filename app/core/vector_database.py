# 벡터DB 관리(chromaDB 연동)

# 로컬 ChromaDB 필요 임포트
import os

import chromadb

# 현재 파일 기준으로 루트 디렉토리 경로 계산
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
chroma_path = os.path.join(BASE_DIR, "chroma_db")
# 로컬 ChromaDB 클라이언트 및 컬렉션 초기화
chroma_client = chromadb.PersistentClient(path=chroma_path)
"""

# 서버 모드에서 ChromaDB에 연결
chroma_client = chromadb.HttpClient(host="http://localhost:8001")

"""

user_collection = chroma_client.get_or_create_collection("user_profiles")
similarity_collection = chroma_client.get_or_create_collection("user_similarities")


async def get_user_data(user_id: str):
    return user_collection.get(ids=[user_id], include=["metadatas"])


async def get_users_data(user_ids: list[str]):
    return user_collection.get(ids=user_ids, include=["metadatas"])


async def get_user_similarities(user_id: str):
    return similarity_collection.get(ids=user_id, include=["metadatas"])


# 사용자 리스트 내부 확인용
async def list_users():
    return user_collection.get()


# 유사도 리스트 내부 확인용
async def list_similarities():
    return similarity_collection.get()
