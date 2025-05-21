import logging
import os

import chromadb

chroma_client = None


def is_client_alive(client):
    try:
        client.list_collections()
        return True
    except Exception as e:
        logging.warning(f"[Chroma] 클라이언트 응답 없음: {e}")
        return False


def get_chroma_client():
    global chroma_client

    if chroma_client is not None and is_client_alive(chroma_client):
        return chroma_client
    else:
        print("ChromaDB 클라이언트 재연결 시도")
        chroma_client = None  # 죽은 연결 무효화

    try:
        mode = os.getenv("CHROMA_MODE", "local")

        if mode == "local":
            chroma_path = os.getenv("CHROMA_PATH")
            if not chroma_path:
                raise RuntimeError("CHROMA_PATH 환경변수가 설정되지 않았습니다.")

            chroma_client = chromadb.PersistentClient(path=chroma_path)
        else:
            host = (
                os.getenv("CHROMA_HOST", "localhost")
                .replace("http://", "")
                .replace("https://", "")
            )
            port = int(os.getenv("CHROMA_PORT", "8001"))
            chroma_client = chromadb.HttpClient(host=host, port=port)

        if not is_client_alive(chroma_client):
            logging.error("ChromaDB 클라이언트가 연결되었지만 응답이 없습니다.")
            raise RuntimeError("ChromaDB 클라이언트가 연결되었지만 응답이 없습니다.")

        return chroma_client
    except Exception as e:
        logging.exception(f"[Chroma] 클라이언트 초기화 실패: {e}")
        chroma_client = None
        return None
