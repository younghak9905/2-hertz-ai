# app/models/qwen_loader.py

import os
import time
from typing import Any, Dict, List

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv(override=True)


class QwenLoader:
    """
    Qwen 모델을 로드하고 관리하는 클래스
    GCP VM 인스턴스에서 직접 GPU를 사용하도록 구현
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct-1M"):
        """
        Qwen 모델 및 토크나이저 초기화

        Args:
            model_name: 사용할 Qwen 모델 이름
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        start_time = time.time()
        print(f"Loading tokenizer from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"Loading model from {model_name}...")
        model_kwargs = {
            "torch_dtype": (
                torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )
        }
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", **model_kwargs  # 자동으로 GPU에 메모리 할당
        )

        # 토큰 생성 파라미터 설정
        self.generation_config = {
            "temperature": 0.3,  # 낮은 온도값으로 일관된 결과 생성
            "top_p": 0.9,
            "max_new_tokens": 1024,  # 최대 생성 토큰 수
            "do_sample": True,
        }

        load_time = time.time() - start_time
        print(f"Model loading completed in {load_time:.2f} seconds")

        # 모델 예열 (첫 추론 시간 단축)
        self._warmup()

    def _warmup(self):
        """모델 예열을 위한 간단한 추론 실행"""
        print("Warming up the model...")
        start_time = time.time()

        warmup_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ]

        # 예열용 추론 실행
        self.get_response(warmup_messages)

        warmup_time = time.time() - start_time
        print(f"Model warmup completed in {warmup_time:.2f} seconds")

    def get_response(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        메시지 목록을 기반으로 모델에서 응답 생성

        Args:
            messages: system, user, assistant 메시지 목록

        Returns:
            생성된 응답을 포함한 딕셔너리
        """
        start_time = time.time()

        try:
            # 채팅 템플릿 적용
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # 입력 토큰화
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # 모델 추론 실행
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **self.generation_config,
                    stopping_criteria=None,  # 사용 가능하면 here we can add stopping criteria
                )

            # 생성된 텍스트 디코딩
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

            end_time = time.time()
            print(f"Inference completed in {end_time - start_time:.2f} seconds")

            return {
                "status_code": 200,
                "content": generated_text.strip(),
                "inference_time": end_time - start_time,
            }

        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return {"status_code": 500, "error": str(e)}


# 싱글톤 인스턴스 생성
_qwen_instance = None


def get_model():
    """
    Qwen 모델 싱글톤 인스턴스 반환
    처음 호출 시에만 모델을 로드하고, 이후 호출에서는 이미 로드된 인스턴스 반환

    Returns:
        QwenLoader: 초기화된 Qwen 모델 인스턴스
    """
    global _qwen_instance

    if _qwen_instance is None:
        # 환경 변수에서 모델 이름 가져오기 (기본값: "Qwen/Qwen2.5-7B-Instruct-1M")
        model_name = os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct-1M")
        _qwen_instance = QwenLoader(model_name)

    return _qwen_instance
