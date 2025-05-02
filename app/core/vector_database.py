# 벡터DB 관리(chromaDB 연동)

import os

import chromadb

# 현재 파일 기준으로 루트 디렉토리 경로 계산
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
chroma_path = os.path.join(BASE_DIR, "chroma_db")

# ChromaDB 클라이언트 및 컬렉션 초기화
chroma_client = chromadb.PersistentClient(path=chroma_path)


user_collection = chroma_client.get_or_create_collection("user_profiles")
similarity_collection = chroma_client.get_or_create_collection("user_similarities")


# --- 추후에 필요한 항목만 가져오도록 수정하기


# 사용자 리스트 내부 확인용
async def list_users():
    return user_collection.get()


# 유사도 리스트 내부 확인용
async def list_similarities():
    return similarity_collection.get()
