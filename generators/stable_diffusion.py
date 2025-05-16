from diffusers import DiffusionPipeline
from PIL import Image
import torch

class StableDiffusion2:
    def __init__(self, model_path: str = None):
        self.pipe = DiffusionPipeline.from_pretrained(model_path if model_path else "stabilityai/stable-diffusion-2-1")
        self.pipe.to("cuda")

    def generate_image(self, prompt: str, generator: torch.Generator) -> Image.Image:
        result = self.pipe(prompt)
        return result.images[0]
    