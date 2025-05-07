# -*- coding: utf-8 -*-
"""
헬스 체크 API 라우터
시스템 및 외부 서비스 연결 상태 확인을 위한 엔드포인트 제공
"""
import os

import chromadb
from fastapi import APIRouter


class HealthRouter:
    """
    서비스 상태 확인을 위한 헬스 체크 라우터
    """

    def __init__(self):
        self.router = APIRouter(prefix="/api/v1/health", tags=["Health"])
        self._configure_routes()
        self._init_chroma_client()

    def _init_chroma_client(self):
        """ChromaDB 클라이언트 초기화"""
        try:
            chroma_host = os.getenv("CHROMA_HOST", "localhost")
            chroma_port = os.getenv("CHROMA_PORT", "8001")
            self.chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        except Exception as e:
            print(f"ChromaDB 클라이언트 초기화 실패: {e}")
            # 대체 클라이언트 시도
            try:
                self.chroma_client = chromadb.Client()
            except Exception as e:
                print(f"ChromaDB 기본 클라이언트 초기화 실패: {e}")
                self.chroma_client = None

    def _configure_routes(self):
        """라우터 경로 설정"""
        self.router.add_api_route("", self.check_health, methods=["GET"])
        self.router.add_api_route("/chromadb", self.check_chromadb, methods=["GET"])

    async def check_health(self):
        """기본 헬스 체크 엔드포인트"""
        return {"status": "UP", "message": "서비스가 정상적으로 실행 중입니다"}

    async def check_chromadb(self):
        """ChromaDB 연결 확인 엔드포인트"""
        if self.chroma_client is None:
            return {
                "status": "DOWN",
                "message": "ChromaDB 클라이언트 초기화에 실패했습니다",
            }

        try:
            # ChromaDB 연결 확인 (컬렉션 목록 가져오기)
            collections = self.chroma_client.list_collections()
            return {
                "status": "UP",
                "message": "ChromaDB 연결 정상",
                "collections_count": len(collections),
            }
        except Exception as e:
            return {"status": "DOWN", "message": f"ChromaDB 연결 실패: {str(e)}"}
