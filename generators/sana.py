import torch
from diffusers import SanaSprintPipeline
from PIL import Image
import random

class Sana:
    def __init__(self, model_path: str = None):
        self.pipe = SanaSprintPipeline.from_pretrained(model_path if model_path else "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
            torch_dtype=torch.bfloat16
        )
        self.pipe.to("cuda")

    def generate_image(self, prompt: str, generator: torch.Generator) -> Image.Image:
        # Generate the image
        result = self.pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            guidance_scale=4.5,
            num_inference_steps=2,
            generator=generator,
            output_type="pil"
        )

        return result.images[0]