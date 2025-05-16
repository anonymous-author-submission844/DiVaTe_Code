# __init__.py inside generators/

#from .flux import Flux
from .sana import Sana
from .stable_diffusion import StableDiffusion2
from .stable_diffusion3 import StableDiffusion3
from .gpt_image_1 import GPT_Image_1
from .gemini import Gemini
from .textdiffuser2 import TextDiffuser2
from .deepfloyd import DeepFloyd
from .anytext import AnyText
from .flux import Flux
# from .grok import Grok

def load_model(args, phase=None):
    if args.model == "sana":
        return Sana(args.model_path)
    elif args.model == "flux":
        return Flux(args.model_path, args.model_path_2)
    elif args.model == "gemini":
        return Gemini(args.api_key)
    # elif args.model == "grok":
    #    return Grok(args.api_key)
    elif args.model == "gemini":
        return Gemini(args.api_key)
    elif args.model == "stable_diffusion":
        return StableDiffusion2(args.model_path)
    elif args.model == "stable_diffusion3":
        return StableDiffusion3(args.model_path)
    elif args.model == "gpt_image_1":
        return GPT_Image_1(args.api_key)
    elif args.model == "textdiffuser2":
        return TextDiffuser2(args.model_path, args.model_path_2, args.model_path_3, phase=phase)
    elif args.model == "gpt_image_1":
        return GPT_Image_1(args.api_key)
    elif args.model == "textdiffuser2":
        return TextDiffuser2(args.model_path, args.model_path_2, args.model_path_3, phase=phase)
    elif args.model == "deepfloyd":
        return DeepFloyd(args.model_path, args.model_path_2, args.model_path_3)
    elif args.model.lower() == "anytext":
        return AnyText(args.model_path)
    else:
        raise ValueError(f"Unknown model: {args.model}")