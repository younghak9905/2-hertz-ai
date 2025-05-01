# 벡터DB 관리(chromaDB 연동)

import chromadb

# ChromaDB 클라이언트 및 컬렉션 초기화
chroma_client = chromadb.PersistentClient(path="./chroma_db")

user_collection = chroma_client.get_or_create_collection("user_profiles")
similarity_collection = chroma_client.get_or_create_collection("user_similarities")
