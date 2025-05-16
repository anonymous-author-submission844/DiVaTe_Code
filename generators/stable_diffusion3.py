import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image
import torch

class StableDiffusion3:
    def __init__(self, model_path: str = None):
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_path if model_path else "stabilityai/stable-diffusion-3-medium-diffusers", 
            torch_dtype=torch.float16
        )
        self.pipe.to("cuda")

    def generate_image(self, prompt: str, generator: torch.Generator) -> Image.Image:
        result = self.pipe(
            prompt=prompt,
            prompt_3=prompt,
            negative_prompt="",        # You can customize this if needed
            num_inference_steps=28,
            height=1024,
            width=1024,
            guidance_scale=7.0,
            generator=generator
        )
        return result.images[0]