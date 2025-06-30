"""
SBERT 모델 로더 모듈
한국어 문장 임베딩을 위한 SBERT 모델을 효율적으로 로드하고 관리
싱글톤 패턴을 적용하여 메모리 사용량 최적화 및 일관된 추론 환경 제공
"""

import os
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


# 모듈 임포트 시점에 바로 모델 초기화
def _load_model():
    # CPU 스레드 수 최적화 - 시스템의 모든 코어 활용
    torch.set_num_threads(max(1, os.cpu_count() // 2))  # 최소 1개는 사용하도록 보장

    # 모델 로드

    # 기존 코드
    # loaded_model = SentenceTransformer("jhgan/ko-sbert-nli")

    # 환경변수에서 모델 경로 가져오기

    MODEL_NAME = "jinkyeongk/kcELECTRA-toxic-detector"
    MODEL_DIR_NAME = MODEL_NAME.replace("/", "-")

    # app-tuning 디렉토리 기준으로 고정
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # KCELECTRA_BASE_HOME 환경변수 있으면 사용, 없으면 model-cache 기본 경로
    MODEL_CACHE = os.environ.get(
        "KCELECTRA_BASE_HOME", os.path.join(BASE_DIR, "model-cache")
    )

    # 모델 최종 경로
    MODEL_PATH = Path(MODEL_CACHE) / MODEL_DIR_NAME

    # 모델 경로 존재 확인
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"모델 경로가 존재하지 않습니다: {MODEL_PATH}\n"
            f"모델을 다운로드하려면 'scripts/download_model.py'를 실행하세요."
        )
    # 모델 및 토크나이저 로드
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # pipeline 구성
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )

    # 예열 (간단한 추론 한번 수행)
    _ = pipe("모델 예열용 텍스트")

    return pipe


# 모듈 레벨에서 모델 초기화
model = _load_model()


# 모델 인스턴스에 접근하기 위한 간단한 함수
def get_model():
    """
    초기화된 SBERT 모델 인스턴스 반환

    Returns:
        SentenceTransformer: 초기화된 한국어 SBERT 모델 인스턴스
    """
    return model
