import torch    
from diffusers import FluxPipeline, FluxTransformer2DModel
from transformers import T5EncoderModel
from optimum.quanto import freeze, qfloat8, quantize
from PIL import Image
import random
import torch
class Flux:
    def __init__(self, model_path: str = None, model_path_2: str = None):
        transformer = FluxTransformer2DModel.from_single_file(
            model_path,
            torch_dtype=torch.bfloat16,
        )
        quantize(transformer, weights=qfloat8)
        freeze(transformer)

        text_encoder_2 = T5EncoderModel.from_pretrained(model_path_2, subfolder="text_encoder_2", torch_dtype=torch.bfloat16)
        quantize(text_encoder_2, weights=qfloat8)
        freeze(text_encoder_2)

        pipe = FluxPipeline.from_pretrained(model_path_2, transformer=None, text_encoder_2=None, torch_dtype=torch.bfloat16)
        pipe.transformer = transformer
        pipe.text_encoder_2 = text_encoder_2

        self.pipe = pipe
        self.pipe.to("cuda")

    def generate_image(self, prompt: str, generator: torch.Generator) -> Image.Image:
        images = self.pipe(
            prompt=prompt,
            prompt_2=prompt,
            height=448,
            width=448,
            guidance_scale=12.0,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=generator
        ).images
        return images[0]
        