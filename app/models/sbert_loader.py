"""
SBERT 모델 로더 모듈
한국어 문장 임베딩을 위한 SBERT 모델을 효율적으로 로드하고 관리
싱글톤 패턴을 적용하여 메모리 사용량 최적화 및 일관된 추론 환경 제공
"""

import os

import torch
from sentence_transformers import SentenceTransformer

# import threading


# ----- 기존 코드 ----- #

# model = SentenceTransformer("jhgan/ko-sbert-nli")

# ----- 기존 코드 ----- #


# ----- 1차 수정 코드 ----- #

# # 모델 싱글톤 인스턴스
# _model = None
# model_lock = threading.Lock()


# def get_model():
#     """
#     SBERT 모델 싱글톤 인스턴스 반환

#     처음 호출 시에만 모델을 로드하고, 이후 호출에서는 이미 로드된 인스턴스 반환
#     스레드 안전한 지연 초기화(lazy initialization) 구현

#     Returns:
#         SentenceTransformer: 초기화된 한국어 SBERT 모델 인스턴스
#     """
#     global _model

#     if _model is not None:
#         return _model

#     with model_lock:
#         if _model is not None:
#             return _model

#         # CPU 스레드 수 최적화
#         torch.set_num_threads(min(4, os.cpu_count() or 4))

#         # 모델 로드
#         _model = SentenceTransformer("jhgan/ko-sbert-nli")

#         # 가능하면 반정밀도(FP16) 사용
#         if torch.cuda.is_available():
#             _model = _model.half().to("cuda")
#         else:
#             # CPU 최적화
#             _model = _model.to("cpu")

#         # 모델 예열 (첫 추론 시간 단축)
#         _ = _model.encode("모델 예열용 텍스트")

#         return _model


# # 편의를 위한 model 변수 제공 (기존 코드 호환성)
# model = get_model()

# ----- 1차 수정 코드 ----- #


# ----- 2차 수정 코드 ----- #


# 모듈 임포트 시점에 바로 모델 초기화
def _load_model():
    # CPU 스레드 수 최적화 - 시스템의 모든 코어 활용
    torch.set_num_threads(os.cpu_count() or 4)  # 제한 없이 모든 코어 사용

    # 모델 로드
    loaded_model = SentenceTransformer("jhgan/ko-sbert-nli")

    # GPU가 있는 경우에만 GPU로 이동
    if torch.cuda.is_available():
        loaded_model = loaded_model.half().to("cuda")

    # 모델 예열 (첫 추론 시간 단축)
    _ = loaded_model.encode("모델 예열용 텍스트")

    return loaded_model


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


# ----- 2차 수정 코드 ----- #
