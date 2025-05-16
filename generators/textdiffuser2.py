import string
from transformers import (
    CLIPTokenizer,
    CLIPTextModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
)
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
import torch
from PIL import Image
from fastchat.model import get_conversation_template
from diffusers.utils.import_utils import is_xformers_available
import xformers

class TextDiffuser2:
    def __init__(self, 
                 base_model_path="stable-diffusion-v1-5/stable-diffusion-v1-5",
                 model_path="JingyeChen22/textdiffuser2-full-ft",
                 layout_model_path="JingyeChen22/textdiffuser2_layout_planner",
                 phase="layout"):
        
        self.phase = phase
        if phase == "layout":
            # Load only layout model
            self.layout_tokenizer = AutoTokenizer.from_pretrained(layout_model_path, use_fast=False)
            layout_config = AutoConfig.from_pretrained(layout_model_path)
            if isinstance(layout_config.torch_dtype, str):
                layout_config.torch_dtype = torch.float16
            self.layout_model = AutoModelForCausalLM.from_pretrained(
                layout_model_path, config=layout_config, torch_dtype=torch.float16
            ).cuda()
            self.layout_model.eval()
        else:  # phase == "image"
            # Load tokenizer and extend with special position/char tokens
            self.tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
            alphabet = string.digits + string.ascii_lowercase + string.ascii_uppercase + string.punctuation + ' '
            for i in range(520):
                self.tokenizer.add_tokens([f"l{i}", f"t{i}", f"r{i}", f"b{i}"])
            for c in alphabet:
                self.tokenizer.add_tokens([f"[{c}]"])

            self.text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder").half().cuda()
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))

            self.vae = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae").half().cuda()
            self.unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet").half().cuda()
            self.unet.enable_xformers_memory_efficient_attention()
            self.scheduler = DDPMScheduler.from_pretrained(base_model_path, subfolder="scheduler")

    def _generate_layout(self, prompt: str) -> str:
        if self.phase != "layout":
            raise ValueError("Layout generation is only available in layout phase")
            
        template = (
            f"Given a prompt that will be used to generate an image, plan the layout of visual text for the image. "
            f"The size of the image is 128x128. Therefore, all properties of the positions should not exceed 128, "
            f"including the coordinates of top, left, right, and bottom. All keywords are included in the caption. "
            f"You don't need to specify the details of font styles. At each line, the format should be keyword left, top, right, bottom. "
            f"So let us begin. Prompt: {prompt}"
        )
        conv = get_conversation_template(self.layout_model.name_or_path)
        conv.append_message(conv.roles[0], template)
        conv.append_message(conv.roles[1], None)
        lm_input = conv.get_prompt()

        input_ids = self.layout_tokenizer([lm_input], return_token_type_ids=False)
        input_ids = {k: torch.tensor(v).cuda() for k, v in input_ids.items()}

        with torch.no_grad():
            output_ids = self.layout_model.generate(
                **input_ids,
                do_sample=True,
                temperature=0.9,
                top_p=0.6,
                repetition_penalty=1.0,
                max_new_tokens=512,
            )[0]
        output_ids = output_ids[len(input_ids["input_ids"][0]):]
        return self.layout_tokenizer.decode(output_ids, skip_special_tokens=True)

    def _compose_ids(self, prompt: str, layout_str: str):
        if self.phase != "image":
            raise ValueError("Image generation is only available in image phase")
            
        layout_lines = layout_str.strip().split("\n")
        layout_tokens = []
        for line in layout_lines:
            if not line or '###' in line or '.com' in line:
                continue
            *chars, box = line.strip().split()
            l, t, r, b = map(int, box.split(','))
            layout_tokens += [f"l{l}", f"t{t}", f"r{r}", f"b{b}"] + [f"[{c}]" for c in " ".join(chars)] + [self.tokenizer.eos_token_id]

        prompt_ids = self.tokenizer(prompt, truncation=True, return_tensors="pt").input_ids[0].tolist()
        try:
            layout_ids = self.tokenizer.encode(layout_tokens)
            full_ids = prompt_ids + layout_ids
        except:
            full_ids = prompt_ids

        full_ids = full_ids[:77] + [self.tokenizer.pad_token_id] * (77 - len(full_ids))
        return torch.tensor([full_ids], device="cuda")

    def generate_image(self, prompt: str, generator: torch.Generator, layout_str: str = None, steps=20, guidance_scale=7.5) -> Image.Image:
        if self.phase == "layout":
            return self._generate_layout(prompt)
        else:  # phase == "image"
            if layout_str is None:
                raise ValueError("layout_str must be provided in image phase")
                
            input_ids = self._compose_ids(prompt, layout_str)
            uncond_ids = torch.tensor([[self.tokenizer.pad_token_id]*77], dtype=torch.long, device="cuda")

            self.scheduler.set_timesteps(steps)
            latent = torch.randn((1, 4, 64, 64), generator=generator, device="cuda", dtype=torch.float16)

            cond_hidden = self.text_encoder(input_ids)[0]
            uncond_hidden = self.text_encoder(uncond_ids)[0]

            for t in self.scheduler.timesteps:
                noise_cond = self.unet(latent, t, encoder_hidden_states=cond_hidden).sample
                noise_uncond = self.unet(latent, t, encoder_hidden_states=uncond_hidden).sample
                noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                latent = self.scheduler.step(noise, t, latent).prev_sample

            latent = 1 / self.vae.config.scaling_factor * latent
            decoded = self.vae.decode(latent).sample[0].detach().float().cpu()
            decoded = (decoded / 2 + 0.5).clamp(0, 1).permute(1, 2, 0)
            return Image.fromarray((decoded.numpy() * 255).astype("uint8"))
