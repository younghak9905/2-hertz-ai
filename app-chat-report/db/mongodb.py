import os
from typing import Any, Dict

import certifi
from dotenv import load_dotenv
from fastapi import HTTPException
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from utils.logger import logger

load_dotenv()  # .env 파일 자동 로드


env = os.getenv("ENVIRONMENT")
if env == "dev":
    DB_NAME = os.getenv("MONGODB_DB_NAME_DEV")
else:
    DB_NAME = os.getenv("MONGODB_DB_NAME")

MONGO_URI = os.getenv("MONGODB_CONNECTION_STRING")
if not MONGO_URI:
    raise ValueError(
        "MONGODB_CONNECTION_STRING 환경 변수가 없습니다. .env 파일을 확인해주세요."
    )


class MongoDB:
    def __init__(self):
        try:
            # 앱 시작 시 client를 초기화
            self.client = MongoClient(
                MONGO_URI,
                tlsCAFile=certifi.where(),
                serverSelectionTimeoutMS=5000,
            )
            self.client.admin.command("ping")
            self.db = self.client.get_database(DB_NAME)
            self.collection = self.db.get_collection("chat_reports")
            logger.info("🟢 MongoDB에 성공적으로 연결되었고, 헬스체크를 통과했습니다.")

        except ConnectionFailure as e:
            # 4. 연결 실패 시 에러 로깅 및 프로세스 종료
            logger.error(f"🔴 MongoDB 연결 실패: 헬스체크에 실패했습니다. 에러: {e}")
            raise  # 에러를 다시 발생시켜 FastAPI 시작을 중단

    def get_collection(self):
        # 서비스 파일에서 사용할 컬렉션 객체를 반환하는 함수
        return self.collection

    def close(self):
        # 앱 종료 시 연결을 닫음
        self.client.close()
        logger.info("⚪ MongoDB 연결이 해제되었습니다.")


# 전역 변수로 MongoDB 인스턴스 생성
mongodb = MongoDB()


# =================================================================
# 데이터베이스 함수
# =================================================================
def save_report_to_db(report_data: Dict[str, Any]) -> None:
    """MongoDB에 신고 데이터 저장"""
    try:
        chat_report_collection = mongodb.get_collection()
        chat_report_collection.insert_one(report_data)
        logger.info(
            f"Chat report saved: {report_data['messageId']}, "
            f"Result: {report_data['result']}, "
            f"Label: {report_data['label']}, "
            f"Confidence: {report_data['confidence']}"
        )
    except Exception as e:
        logger.error(f"MongoDB save failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
