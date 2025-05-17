import os
import time

import requests
from dotenv import load_dotenv


# 튜닝리포트(뉴스) 생성 모델 로드
class QwenLoader:
    def __init__(self, mode="colab"):
        self.mode = mode
        self.model_path = "Qwen/Qwen3-8B"
        # colab/ngrok API 요청에 필요한 헤더 및 데이터
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer dummy-key",
        }
        self.data = {
            "model": self.model_path,
            "messages": [],
            "temperature": 0.8,
            "top_p": 0.9,
            "max_tokens": 1024,
            "stop": ["\n\n", "</s>"],
        }

        # GCP(vllm) 모드일 때만 vllm 엔진 초기화
        if self.mode == "gcp":
            from vllm import LLM, SamplingParams

            self.model_vllm = LLM(
                model=self.model_path,
                dtype="half",  # 또는 torch.bfloat16, torch.float16 등. torch.bfloat16은 최신 GPU(Ampere 이상)에서만 지원됨.
                trust_remote_code=True,
                tensor_parallel_size=1,
                max_model_len=8192,
                gpu_memory_utilization=0.9,
                max_num_seqs=8,
                max_num_batched_tokens=4096,
            )

            self.sampling_params = SamplingParams(
                temperature=0.0, top_p=0.9, max_tokens=1024, stop=["\n\n", "</s>"]
            )

    def get_response(self, messages):
        if self.mode == "colab":
            from transformers import AutoTokenizer

            load_dotenv(override=True)
            base_url = os.getenv("NGROK_URL")
            url = f"{base_url}/v1/completions"

            # messages → 단일 prompt 텍스트로 변환
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            payload = {
                "model": self.model_path,
                "prompt": prompt,
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": 1024,
                "stop": ["}"],
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer dummy-key",
            }

            start_time = time.time()
            response = requests.post(url, headers=headers, json=payload)
            end_time = time.time()
            print(f"response time: {end_time - start_time:.2f}s")

            if response.status_code != 200:
                return {
                    "status_code": response.status_code,
                    "url": response.url,
                    "error": response.text,
                }

            body = response.json()
            return {
                "status_code": 200,
                "url": response.url,
                "content": body["choices"][0]["text"],
            }


_qwen_instance = QwenLoader(mode="colab")  # 또는 "gcp"로 바꾸세요


def get_model():
    return _qwen_instance
