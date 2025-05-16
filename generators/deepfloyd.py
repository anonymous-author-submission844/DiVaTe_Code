import torch
from PIL import Image
from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch


class DeepFloyd:
    def __init__(self, model_path_1: str, model_path_2: str, model_path_3: str):
        self.model_path_1 = model_path_1
        self.model_path_2 = model_path_2
        self.model_path_3 = model_path_3
        
        # stage 1
        self.stage_1 = DiffusionPipeline.from_pretrained(self.model_path_1 if self.model_path_1 else "DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
        self.stage_1.to("cuda")
        # self.stage_1.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
        # self.stage_1.enable_model_cpu_offload()

        # stage 2
        self.stage_2 = DiffusionPipeline.from_pretrained(
            self.model_path_2 if self.model_path_2 else "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
        )
        self.stage_2.to("cuda")
        # self.stage_2.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
        # self.stage_2.enable_model_cpu_offload()

        # stage 3
        # safety_modules = {"feature_extractor": self.stage_1.feature_extractor, "safety_checker": self.stage_1.safety_checker} #, "watermarker": self.stage_1.watermarker}
        self.stage_3 = DiffusionPipeline.from_pretrained(self.model_path_3 if self.model_path_3 else "stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16)
        self.stage_3.to("cuda")
        # self.stage_3.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
        # self.stage_3.enable_model_cpu_offload()

    def generate_image(self, prompt: str, generator: torch.Generator, steps=20, guidance_scale=7.5) -> Image.Image:
        # text embeds
        prompt_embeds, negative_embeds = self.stage_1.encode_prompt(prompt)

        image = self.stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt").images
        image = self.stage_2(
            image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt"
        ).images

        image = self.stage_3(prompt=prompt, image=image, generator=generator, noise_level=100)
        
        return image.images[0]