# 벡터DB 관리(chromaDB 연동)

# 로컬 ChromaDB 필요 임포트
import os

import chromadb

chroma_client = None
user_collection = None
similarity_collection = None


"""
    # 현재 파일 기준으로 루트 디렉토리 경로 계산
    BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    chroma_path = os.path.join(BASE_DIR, "chroma_db")

    # 로컬 ChromaDB 클라이언트 및 컬렉션 초기화
    chroma_client = chromadb.PersistentClient(path=chroma_path)

    # 서버 모드에서 ChromaDB에 연결
    # chroma_client = chromadb.HttpClient(host="http://localhost:8001")

    # user_collection = chroma_client.get_or_create_collection("user_profiles")
    # similarity_collection = chroma_client.get_or_create_collection("user_similarities")
"""


def get_chroma_client():
    global chroma_client
    if chroma_client is not None:
        return chroma_client

    try:
        mode = os.getenv("CHROMA_MODE", "server")  # "local" 또는 "server"

        if mode == "local":
            # 로컬 PersistentClient 사용
            base_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            chroma_path = os.path.join(base_dir, "chroma_db")
            chroma_client = chromadb.PersistentClient(path=chroma_path)

        else:
            # 서버 모드 (기본)
            host = os.getenv("CHROMA_HOST", "localhost")
            port = int(os.getenv("CHROMA_PORT", "8001"))

            if not host.startswith("http"):
                host = f"http://{host}"

            chroma_client = chromadb.HttpClient(host=host, port=port)

        return chroma_client

    except Exception as e:
        print(f"[ChromaDB 클라이언트 초기화 실패] {e}")
        chroma_client = None
        return None


def get_user_collection():
    global user_collection
    if user_collection is not None:
        return user_collection

    client = get_chroma_client()
    if client is None:
        raise RuntimeError("ChromaDB 클라이언트를 사용할 수 없습니다.")

    try:
        user_collection = client.get_or_create_collection("user_profiles")
        return user_collection
    except Exception as e:
        raise RuntimeError(f"user_profiles 컬렉션 초기화 실패: {e}")


def get_similarity_collection():
    global similarity_collection
    if similarity_collection is not None:
        return similarity_collection

    client = get_chroma_client()
    if client is None:
        raise RuntimeError("ChromaDB 클라이언트를 사용할 수 없습니다.")

    try:
        similarity_collection = client.get_or_create_collection("user_similarities")
        return similarity_collection
    except Exception as e:
        raise RuntimeError(f"user_similarities 컬렉션 초기화 실패: {e}")


def get_user_data(user_id: str):
    collection = get_user_collection()
    return collection.get(ids=[user_id], include=["metadatas"])


async def get_users_data(user_ids: list[str]):
    collection = get_user_collection()
    return collection.get(ids=user_ids, include=["metadatas"])


async def get_user_similarities(user_id: str):
    collection = get_similarity_collection()
    return collection.get(ids=user_id, include=["metadatas"])


async def list_users():
    collection = get_user_collection()
    return collection.get()


async def list_similarities():
    collection = get_similarity_collection()
    return collection.get()
