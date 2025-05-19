# 벡터DB 관리(chromaDB 연동)

# 로컬 ChromaDB 필요 임포트
import json
import os
from http.client import HTTPException

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

    def is_client_alive(client):
        try:
            client.list_collections()  # 헬스체크
            return True
        except Exception as e:
            print(f"ChromaDB 클라이언트 헬스체크 실패: {e}")
            return False

    if chroma_client is not None:
        if is_client_alive(chroma_client):
            return chroma_client
        else:
            print("ChromaDB 클라이언트 재연결 시도")
            chroma_client = None  # 죽은 연결 무효화

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

            host = host.replace("http://", "").replace("https://", "")

            chroma_client = chromadb.HttpClient(host=host, port=port)

        # 연결 테스트
        if not is_client_alive(chroma_client):
            raise RuntimeError("ChromaDB 클라이언트가 연결되었지만 응답이 없습니다.")

        return chroma_client

    except Exception:
        chroma_client = None
        return None


def check_database():
    try:
        client = get_chroma_client()
        if client is None:
            raise RuntimeError("ChromaDB 클라이언트를 사용할 수 없습니다.")

        # 전역 캐시 초기화
        global user_collection, similarity_collection
        user_collection = client.get_or_create_collection("user_profiles")
        similarity_collection = client.get_or_create_collection("user_similarities")

    except Exception as e:
        raise RuntimeError(f"ChromaDB 컬렉션 초기화 실패: {e}")


def get_user_collection():
    global user_collection
    try:
        if user_collection is not None:
            #  유효성 검사
            try:
                user_collection.count()  # 또는 get(ids=[]) 등
                return user_collection
            except Exception:
                user_collection = None  # 죽은 연결 제거

        # 클라이언트 연결
        client = get_chroma_client()
        if client is None:
            raise RuntimeError("ChromaDB 클라이언트를 사용할 수 없습니다.")

        user_collection = client.get_or_create_collection("user_profiles")
        return user_collection

    except Exception as e:
        print(f" user_profiles 컬렉션 초기화 실패: {e}")
        raise RuntimeError(f"user_profiles 컬렉션 초기화 실패: {e}")


def get_similarity_collection():
    global similarity_collection
    try:
        if similarity_collection is not None:
            try:
                similarity_collection.count()
                return similarity_collection
            except:
                print(" similarity_collection 무효화됨. 재생성 시도.")
                similarity_collection = None

        client = get_chroma_client()
        if client is None:
            raise RuntimeError("ChromaDB 클라이언트를 사용할 수 없습니다.")

        similarity_collection = client.get_or_create_collection("user_similarities")
        print(" similarity_collection 재생성 완료")
        return similarity_collection

    except Exception as e:
        raise RuntimeError(f"user_similarities 컬렉션 초기화 실패: {e}")


def get_user_data(user_id: str):
    try:
        collection = get_user_collection()
        result = collection.get(ids=[user_id], include=["metadatas"])
        if not result["metadatas"] or result["metadatas"][0] is None:
            raise HTTPException(
                status_code=404, detail="사용자 정보를 찾을 수 없습니다."
            )
        return result
    except Exception as e:
        print(f" get_user_data 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))


#  ChromaDB 데이터베이스 컬렉션 삭제 후, 재생성(내부 테스트용)
def reset_collections():
    try:
        client = get_chroma_client()
        if client is None:
            raise RuntimeError("ChromaDB 클라이언트를 사용할 수 없습니다.")

        # 삭제
        client.delete_collection("user_profiles")
        client.delete_collection("user_similarities")

        # 전역 캐시 초기화
        global user_collection, similarity_collection
        user_collection = client.get_or_create_collection("user_profiles")
        similarity_collection = client.get_or_create_collection("user_similarities")

    except Exception as e:
        raise RuntimeError(f"ChromaDB 컬렉션 초기화 실패: {e}")


# 회원 삭제
def delete_user(user_id: int):
    """
    user_profiles와 similarity_collection에서 해당 유저 직접 삭제
    """
    try:
        check_database()
        user_id = str(user_id)

        user_collection.delete(ids=[str(user_id)])
        similarity_collection.delete(ids=[str(user_id)])

    except Exception as e:
        raise RuntimeError(f"ChromaDB 컬렉션 초기화 실패: {e}")


def clean_up_similarity(user_id: int) -> int:
    user_id = str(user_id)
    try:
        check_database()
        all_docs = similarity_collection.get(include=["metadatas"])

        ids = all_docs.get("ids", [])
        metadatas = all_docs.get("metadatas", [])

        if not ids:
            print("⚠️ similarity_collection에는 삭제할 대상이 없습니다.")
            return 0

        updates_made = 0

        for doc_id, metadata in zip(ids, metadatas):
            raw_json = metadata.get("similarities")
            if not raw_json:
                continue

            try:
                similarities = json.loads(raw_json)
            except json.JSONDecodeError as decode_err:
                print(f"⚠️ JSON 파싱 실패 - 문서 ID {doc_id}, 오류: {decode_err}")
                continue

            # 삭제 대상 user_id가 존재하면 제거 및 업데이트
            if user_id in similarities:
                del similarities[user_id]
                metadata["similarities"] = json.dumps(similarities)

                try:
                    similarity_collection.update(ids=[doc_id], metadatas=[metadata])
                    updates_made += 1
                except Exception as update_err:
                    print(f"❌ 문서 ID '{doc_id}' 업데이트 실패: {update_err}")

        print(f"✅ 총 {updates_made}개의 문서에서 user_id '{user_id}' 제거 완료")
        return updates_made

    except Exception as e:
        print(f"❌ similarity_collection 정리 중 전체 오류 발생: {e}")
        raise


def get_user_metadata(user_id: int, user_collection):
    user_id = str(user_id)
    try:
        result = user_collection.get(ids=[user_id], include=["metadatas"])

        if result and result["metadatas"]:
            metadata = result["metadatas"][0]
            print(f"✅ user_id '{user_id}'에 대한 메타데이터: {metadata}")
            return metadata
        else:
            print(f"⚠️ user_id '{user_id}'에 대한 메타데이터를 찾을 수 없습니다.")
            return None

    except Exception as e:
        raise RuntimeError(f"user_id '{user_id}'의 메타데이터 조회 중 오류 발생: {e}")


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
