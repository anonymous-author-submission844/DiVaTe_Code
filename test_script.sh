# test script for the models


category=category1  # category1, category2_1, category2_2, ...
base_dir= # /your/path/to/save/results/
results_dir=${base_dir}/${model_name}
metric=vqa
model_path= # /your/path/to/models/
qwen_model_path= # /your/path/to/models/Qwen2.5-VL-7B-Instruct
api_key= # your api_key


# sana
python3 main.py --model sana --prompt_path make_prompts/${category}.txt --model_path ${model_path}/Sana_Sprint_1.6B_1024px_diffusers --save_path ${results_dir}

# flux
python3 main.py --model flux --prompt_path make_prompts/${category}.txt --model_path ${model_path}/flux/model/flux_fp8/flux1-dev-fp8.safetensors --model_path_2 ${model_path}/flux/model/FLUX.1-dev --save_path ${results_dir}

# stable diffusion
python3 main.py --model stable_diffusion --prompt_path make_prompts/${category}.txt --model_path ${model_path}/stable-diffusion-2-1 --save_path ${results_dir}

# stable diffusion 3
python3 main.py --model stable_diffusion3 --prompt_path make_prompts/${category}.txt --model_path ${model_path}/stable-diffusion-3-medium-diffusers --save_path ${results_dir}

# gpt image 1
python3 main.py --model gpt_image_1 --prompt_path make_prompts/${category}.txt --api_key ${api_key} --save_path ${results_dir}

# gemini
python3 main.py --model gemini --prompt_path make_prompts/${category}.txt --api_key ${api_key} --save_path ${results_dir}

# textdiffuser2
python3 main.py --model textdiffuser2 --prompt_path make_prompts/${category}.txt \
    --model_path ${model_path}/stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --model_path_2 ${model_path}/text-diffuser-2/text-diffuser-2 \
    --layout_model_path ${model_path}/text-diffuser-2/textdiffuser2_layout_planner \
    --save_path ${results_dir}

# deepfloyd
python3 main.py --model deepfloyd --prompt_path make_prompts/${category}.txt \
    --model_path_1 ${model_path}/DeepFloyd/IF-I-XL-v1.0 \
    --model_path_2 ${model_path}/DeepFloyd/IF-II-L-v1.0 \
    --model_path_3 ${model_path}/stabilityai/stable-diffusion-x4-upscaler \
    --save_path ${results_dir}


model_name=sana # flux, sana, stable_diffusion, stable_diffusion3, textdiffuser2, deepfloyd, anytext
# Postprocessing and evaluation

python3 postprocess.py --output_dir ${results_dir}/${category} \
    --ocr_type qwen \
    --model_path ${qwen_model_path}

python3 evaluation.py --output_dir ${results_dir}/${category} \
    --prompt_path make_prompts/${category}.txt \
    --metric ocr

python3 evaluation.py --output_dir ${results_dir}/${category} \
    --prompt_path make_prompts/${category}.txt \
    --metric clipscore

python3 evaluation.py --output_dir ${results_dir}/${category} \
    --prompt_path make_prompts/${category}.txt \
    --vqa_questions_path make_prompts/vqa_${category}.txt \
    --metric vqa \
    --vqa_model_path ${qwen_model_path}