import os
import argparse
from utils.io import load_prompts
from generators import load_model
import torch
import random
import glob
import json
import gc
import json
import gc

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def get_completed_generations(model_dir):
    completed = set()
    pattern = os.path.join(model_dir, "prompt_*_*.png")
    for file_path in glob.glob(pattern):
        filename = os.path.basename(file_path)
        try:
            parts = filename.split('_')
            prompt_idx = int(parts[1])
            iteration = int(parts[2].split('.')[0])
            completed.add((prompt_idx, iteration))
        except (IndexError, ValueError):
            continue
    return completed

def get_failed_prompts(failed_log_path):
    failed = set()
    if os.path.exists(failed_log_path):
        with open(failed_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    prompt_idx, _, iteration = line.strip().split(',')
                    failed.add((int(prompt_idx), int(iteration)))
                except (ValueError, IndexError):
                    continue
    return failed

def main():
    parser = argparse.ArgumentParser(description="Run T2I generation for a specific model")
    parser.add_argument("--model", type=str, default="sana")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_path_2", type=str, default=None)
    parser.add_argument("--model_path_3", type=str, default=None)
    parser.add_argument("--prompt_path", type=str, default="make_prompts/category1.txt")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    prompts = load_prompts(args.prompt_path)
    print(f"Loaded {len(prompts)} prompts from {args.prompt_path}")

    prompt_file_stem = os.path.splitext(os.path.basename(args.prompt_path))[0]
    model_dir = os.path.join(args.save_path, args.model, prompt_file_stem, "generated_images")
    ensure_dir(model_dir)
    failed_log_path = os.path.join(args.save_path, args.model, prompt_file_stem, "failed_prompts.txt")
    layouts_path = os.path.join(args.save_path, args.model, prompt_file_stem, "layouts.json")

    if args.model == "textdiffuser2":
        print("Phase 1: Generating layouts...")
        layout_model = load_model(args, phase="layout")

        layouts = {}
        if args.resume and os.path.exists(layouts_path):
            with open(layouts_path, 'r', encoding='utf-8') as f:
                layouts = json.load(f)
            print(f"Loaded {len(layouts)} existing layouts")

        seed = 2025
        for idx, prompt in enumerate(prompts):

            prompt_idx = str(idx + 1)
            if prompt_idx not in layouts:
                layouts[prompt_idx] = {}

            for i in range(args.num_images):
                seed_idx = str(i + 1)
                if seed_idx in layouts[prompt_idx]:
                    print(f"[SKIP] Layout exists for prompt {prompt_idx}, seed {seed_idx}")
                    continue

                print(f"[Layout {idx+1}/{len(prompts)} | Seed {i+1}] {prompt}")
                try:
                    layout = layout_model.generate_image(prompt, None)
                    layouts[prompt_idx][seed_idx] = layout
                except Exception as e:
                    print(f"[FAIL] Layout generation failed for prompt {idx+1}, seed {i+1}")
                    print(f"Error: {str(e)}")
                    continue

        with open(layouts_path, 'w', encoding='utf-8') as f:
            json.dump(layouts, f, ensure_ascii=False, indent=2)
        print(f"Saved layouts to {layouts_path}")

        del layout_model
        torch.cuda.empty_cache()
        gc.collect()

        print("\nPhase 2: Generating images...")
        image_model = load_model(args, phase="image")

        completed_generations = get_completed_generations(model_dir) if args.resume else set()
        failed_generations = get_failed_prompts(failed_log_path) if args.resume else set()

        for idx, prompt in enumerate(prompts):

            prompt_idx = str(idx + 1)
            if prompt_idx not in layouts:
                print(f"[SKIP] No layouts for prompt {prompt_idx}")
                continue

            for i in range(args.num_images):
                seed_idx = str(i + 1)
                if seed_idx not in layouts[prompt_idx]:
                    print(f"[SKIP] No layout for prompt {prompt_idx}, seed {seed_idx}")
                    continue
                if args.resume and (idx + 1, i + 1) in completed_generations:
                    print(f"[SKIP] Already completed: Prompt {idx+1}, Iteration {i+1}")
                    continue

                generator = torch.Generator(device="cuda").manual_seed(seed + i * 1000)
                print(f"[Prompt {idx+1} | Iteration {i+1}] Seed: {seed + i * 1000}")

                try:
                    image = image_model.generate_image(prompt, generator, layouts[prompt_idx][seed_idx])
                except Exception as e:
                    print(f"[FAIL] Generation failed for prompt {idx+1} (iteration {i+1})")
                    print(f"Error: {str(e)}")
                    with open(failed_log_path, "a", encoding="utf-8") as f:
                        f.write(f"{idx+1},{prompt.strip()},{i+1}\n")
                    continue

                save_path = os.path.join(model_dir, f"prompt_{idx+1:04d}_{i+1}.png")
                image.save(save_path)
                print(f"[SUCCESS] Saved: {save_path}")

    else:
        model = load_model(args)
        print(f"Loaded model: {args.model}")

        completed_generations = get_completed_generations(model_dir) if args.resume else set()
        failed_generations = get_failed_prompts(failed_log_path) if args.resume else set()

        seed = 2025
        for idx, prompt in enumerate(prompts):

            for i in range(args.num_images):
                if args.resume and (idx + 1, i + 1) in completed_generations:
                    print(f"[SKIP] Already completed: Prompt {idx+1}, Iteration {i+1}")
                    continue

                generator = torch.Generator(device="cuda").manual_seed(seed + i * 1000)
                print(f"[Prompt {idx+1} | Iteration {i+1}] Seed: {seed + i * 1000}")

                try:
                    image = model.generate_image(prompt, generator)
                    save_path = os.path.join(model_dir, f"prompt_{idx+1:04d}_{i+1}.png")
                    image.save(save_path)
                    print(f"[SUCCESS] Saved: {save_path}")
                except Exception as e:
                    print(f"[FAIL] Generation failed for prompt {idx+1} (iteration {i+1})")
                    print(f"Error: {str(e)}")
                    with open(failed_log_path, "a", encoding="utf-8") as f:
                        f.write(f"{idx+1},{prompt.strip()},{i+1}\n")
                    continue

if __name__ == "__main__":
    main()