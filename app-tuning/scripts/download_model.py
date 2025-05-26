import os

from sentence_transformers import SentenceTransformer

MODEL_NAME = "jhgan/ko-sbert-nli"
MODEL_CACHE = os.environ.get("SENTENCE_TRANSFORMERS_HOME", "./model-cache")
MODEL_DIR_NAME = MODEL_NAME.replace("/", "-")
MODEL_PATH = os.path.join(MODEL_CACHE, MODEL_DIR_NAME)

os.makedirs(MODEL_CACHE, exist_ok=True)


def _download_model():

    # 모델 디렉토리에 모델 가중치 파일 등이 이미 존재하는지 체크
    if os.path.isdir(MODEL_PATH) and len(os.listdir(MODEL_PATH)) > 0:
        print(
            f":white_check_mark: '{MODEL_NAME}' 모델이 이미 존재합니다: {MODEL_PATH} (다운로드 건너뜀)"
        )
    else:
        try:
            print(
                f":small_red_triangle_down: '{MODEL_NAME}' 모델을 다운로드합니다 (경로: {MODEL_PATH})..."
            )
            model = SentenceTransformer(MODEL_NAME, cache_folder=MODEL_CACHE)
            model.save(MODEL_PATH)
            print(f":white_check_mark: 모델 다운로드 및 저장 완료: {MODEL_PATH}")
        except Exception as e:
            print(f"[ERROR] 모델 다운로드 또는 저장 중 오류 발생: {e}")
            raise


if __name__ == "__main__":
    _download_model()
