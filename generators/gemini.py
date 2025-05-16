from google import genai
from google.genai import types
from google.api_core.exceptions import ServerError
from PIL import Image
from io import BytesIO
import time
import torch

class Gemini:
    def __init__(self, api_key: str = None):
        self.client = genai.Client(api_key=api_key)
        self.request_count = 0
        self.rate_limit = 10  # requests per minute
        self.rate_limit_window = 60  # seconds

    def _wait_for_rate_limit(self):
        self.request_count += 1
        
        if self.request_count >= self.rate_limit:
            print(f"Rate limit reached ({self.rate_limit} requests). Waiting for {self.rate_limit_window} seconds...")
            time.sleep(self.rate_limit_window)
            self.request_count = 0

    def generate_image(self, prompt: str, generator: torch.Generator) -> Image.Image:
        max_retries = 5
        delay = 2  # seconds

        for attempt in range(1, max_retries + 1):
            try:
                self._wait_for_rate_limit()  # Check and wait for rate limit before making request
                
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash-exp-image-generation",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["TEXT", "IMAGE"]
                    )
                )

                candidate = response.candidates[0]

                # Finish Reason 확인 및 로깅
                print(f"Finish reason: {candidate.finish_reason}")
                if candidate.finish_reason == "IMAGE_SAFETY":
                    print(f"Image generation blocked due to safety reasons for prompt: '{prompt}'") # 프롬프트 내용 로깅
                    return None # 안전 문제 시 None 반환
                elif candidate.finish_reason == "RECITATION":
                    print(f"Image generation blocked due to recitation for prompt: '{prompt}'")
                    return None # 리시테이션 문제 시 None 반환
                elif candidate.finish_reason != "STOP" and candidate.finish_reason != "MAX_TOKENS":
                    print(f"Image generation may have failed or was stopped. Reason: {candidate.finish_reason}")
                    return None # 다른 문제로 생성 실패 시 None 반환

                if candidate.content is None:
                    print("candidate.content is None. No content generated.")
                    return None
                
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        return Image.open(BytesIO(part.inline_data.data))

                raise ValueError("No image found in Gemini response.")

            except ServerError as e:
                if "overloaded" in str(e) or "UNAVAILABLE" in str(e):
                    print(f"[Retry {attempt}/{max_retries}] Gemini is overloaded. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    raise
        raise RuntimeError("Failed to generate image after multiple retries due to server overload.")
