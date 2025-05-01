# FastAPI 사용 가이드

## 1. 가상환경 설정

Python 가상환경을 사용하면 프로젝트별로 독립적인 패키지 관리가 가능합니다.

```bash
# 가상환경 생성
python -m venv .venv

# 가상환경 활성화
## Windows
.venv\Scripts\Activate.ps1
## macOS/Linux
source .venv/bin/activate
```

## 2. 필요한 패키지 설치

가상환경이 활성화된 상태에서 필요한 패키지를 설치합니다.

```bash
# 기본 패키지 설치
pip install "fastapi[standard]" uvicorn

# 추가 패키지 설치 (필요에 따라)
pip install python-dotenv pydantic

# 개발 환경용 패키지 (선택사항)
pip install pytest black flake8

# 모든 패키지를 requirements.txt로 저장
pip freeze > requirements.txt

# 패키지 설치
pip install requirements.txt
```

## 3. FastAPI 애플리케이션 실행

애플리케이션을 실행하려면 uvicorn을 사용합니다.

```bash
# 기본 실행(자동 리로드)
uvicorn main:app --reload

# 특정 포트에서 실행
uvicorn main:app --reload --port 8000

# 모든 네트워크 인터페이스에서 접근 가능하게 실행
uvicorn main:app --reload --host 0.0.0.0
```

## 4. 애플리케이션 확인

애플리케이션이 실행되면 다음 URL에서 접근할 수 있습니다:

- 웹 애플리케이션: http://localhost:8000
- API 문서 (Swagger UI): http://localhost:8000/docs
- 대체 API 문서 (ReDoc): http://localhost:8000/redoc
