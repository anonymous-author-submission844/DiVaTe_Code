from openai import OpenAI, BadRequestError, APIConnectionError, InternalServerError
import base64
from PIL import Image
from io import BytesIO
import torch

class GPT_Image_1:
    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key)

    def generate_image(self, prompt: str, generator: torch.Generator) -> Image.Image | None:
        try:
            response = self.client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                quality="medium"
            )

            image_base64 = response.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)
            return Image.open(BytesIO(image_bytes))

        except BadRequestError as e:
            print(f"[BadRequestError] Prompt rejected: {prompt} | Reason: {e}")
        except APIConnectionError as e:
            print(f"[APIConnectionError] Network issue for prompt '{prompt}': {e}")
        except InternalServerError as e:
            print(f"[InternalServerError] Server issue for prompt '{prompt}': {e}")
        except Exception as e:
            print(f"[UnexpectedError] Prompt '{prompt}' failed: {e}")
        
        return None
