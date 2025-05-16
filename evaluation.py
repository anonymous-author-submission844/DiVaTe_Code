import os
import json
import torch
import argparse
import shutil
import ast
from PIL import Image, ImageDraw
from tqdm import tqdm
import cv2
import numpy as np
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from eval_utils.utils import extract_text_from_prompts
from eval_utils.ocr import calculate_ocr_metrics
from eval_utils.clipscore import calculate_clip_score
from eval_utils.vqascore import calculate_vqa_score, load_vqa_questions, calculate_vqa_metrics


def main():
    parser = argparse.ArgumentParser(description="Run evaluation on the generated images")
    parser.add_argument("--output_dir", type=str, required=True, help="Main output directory containing all subdirectories")
    parser.add_argument("--metric", type=str, choices=['ocr', 'clipscore', 'vqa', 'all'], default='ocr', 
                       help="Type of metric to use")
    parser.add_argument("--clip_model_path", type=str, default=None, help="Path to the clip model")
    parser.add_argument("--prompt_path", type=str, default=None, help="Path to the prompt file")
    parser.add_argument("--vqa_model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Path to the VQA model")
    parser.add_argument("--vqa_questions_path", type=str, default=None, help="Path to a single VQA question file")
    args = parser.parse_args()
    
    print("\n\n")
    print(args.output_dir)
    print(args.prompt_path)
    print("\n\n")
    # Create evaluation results directory
    eval_dir = os.path.join(args.output_dir, "evaluation_results")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Set results file paths
    results_file = os.path.join(eval_dir, "evaluation_results.txt")
    per_prompt_file = os.path.join(eval_dir, "per_prompt_results.txt")
    attribute_results_file = os.path.join(eval_dir, "attribute_evaluation_results.txt")
    per_attribute_file = os.path.join(eval_dir, "per_attribute_results.txt")
    
    # Set CLIP score results file paths
    clip_results_file = os.path.join(eval_dir, "clip_evaluation_results.txt")
    clip_per_prompt_file = os.path.join(eval_dir, "clip_per_prompt_results.txt")
    
    # Set OCR results file paths
    ocr_results_file = os.path.join(eval_dir, "ocr_evaluation_results.txt")
    ocr_per_prompt_file = os.path.join(eval_dir, "ocr_per_prompt_results.txt")
    
    # Load prompts and extract text
    prompt_texts, visual_attributes, match_counts = extract_text_from_prompts(args.prompt_path)
    
    # Paths
    ocr_results_path = os.path.join(args.output_dir, "ocr_results", "ocr_results.json")
    masked_images_path = os.path.join(args.output_dir, "masked_images")
    
    # Initialize results file - only open once in write mode
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=== Evaluation Results ===\n\n")
        
    # Calculate OCR metrics
    if args.metric in ['ocr', 'all']:
        # Calculate metrics for prompt texts
        ocr_metrics, per_prompt_metrics = calculate_ocr_metrics(ocr_results_path, prompt_texts, match_counts)
        print("\nOCR Evaluation Metrics (Prompt Texts):")
        print(f"Accuracy:        {ocr_metrics['accuracy']:.4f}")
        print(f"Edit Distance:   {ocr_metrics['edit_distance']:.4f}")
        
        # Append OCR results to the main results file
        with open(results_file, 'a', encoding='utf-8') as f:
            f.write("=== OCR Evaluation Metrics (Prompt Texts) ===\n")
            f.write(f"Accuracy:        {ocr_metrics['accuracy']:.4f}\n")
            f.write(f"Edit Distance:   {ocr_metrics['edit_distance']:.4f}\n\n")
        
        # Save OCR results to separate file
        with open(ocr_results_file, 'w', encoding='utf-8') as of:
            of.write("=== OCR Evaluation Metrics (Prompt Texts) ===\n")
            of.write(f"Accuracy:        {ocr_metrics['accuracy']:.4f}\n")
            of.write(f"Edit Distance:   {ocr_metrics['edit_distance']:.4f}\n\n")
            of.write("=== OCR Evaluation Metrics (Visual Attributes) ===\n")
            
            # Calculate metrics for visual attributes
            attr_metrics, per_attribute_metrics = calculate_ocr_metrics(ocr_results_path, visual_attributes)
            of.write(f"Accuracy:        {attr_metrics['accuracy']:.4f}\n")
            of.write(f"Edit Distance:   {attr_metrics['edit_distance']:.4f}\n\n")
        
        # Save per-prompt OCR results 
        with open(ocr_per_prompt_file, 'w', encoding='utf-8') as pf:
            for prompt_idx in sorted(per_prompt_metrics.keys()):
                metrics = per_prompt_metrics[prompt_idx]
                pf.write(f"Prompt {prompt_idx + 1}:\n")
                pf.write(f"Text: {prompt_texts[prompt_idx]}\n")
                pf.write(f"Number of matches: {match_counts[prompt_idx]}\n")
                pf.write(f"Accuracy:        {metrics['accuracy']:.4f}\n")
                pf.write(f"Edit Distance:   {metrics['edit_distance']:.4f}\n\n")
        
        # Print visual attribute metrics
        print("\nOCR Evaluation Metrics (Visual Attributes):")
        print(f"Accuracy:        {attr_metrics['accuracy']:.4f}")
        print(f"Edit Distance:   {attr_metrics['edit_distance']:.4f}")
        
        # Save attribute results
        with open(attribute_results_file, 'w', encoding='utf-8') as af:
            af.write(f"Accuracy:        {attr_metrics['accuracy']:.4f}\n")
            af.write(f"Edit Distance:   {attr_metrics['edit_distance']:.4f}\n\n")
        
        # Save per-attribute results
        with open(per_attribute_file, 'w', encoding='utf-8') as af:
            for prompt_idx in sorted(per_attribute_metrics.keys()):
                metrics = per_attribute_metrics[prompt_idx]
                af.write(f"Prompt {prompt_idx + 1}:\n")
                af.write(f"Attribute: {visual_attributes[prompt_idx]}\n")
                af.write(f"Accuracy:        {metrics['accuracy']:.4f}\n")
                af.write(f"Edit Distance:   {metrics['edit_distance']:.4f}\n\n")
                
        print(f"OCR results saved to: {ocr_results_file}")
        print(f"Per-prompt OCR results saved to: {ocr_per_prompt_file}")
    
    # Calculate CLIP score
    if args.metric in ['clipscore', 'all']:
        # Initialize CLIP model
        model_name = args.clip_model_path if args.clip_model_path else "openai/clip-vit-base-patch32"
        clip_model = CLIPModel.from_pretrained(model_name)
        clip_processor = CLIPProcessor.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            clip_model = clip_model.cuda()
            
        clip_metrics, per_prompt_clip = calculate_clip_score(masked_images_path, args.prompt_path, clip_model, clip_processor)
        print(f"\nCLIP Scores:")
        print(f"Full Prompt Score: {clip_metrics['full_prompt_score']:.4f} ± {clip_metrics['full_prompt_std']:.4f}")
        print(f"Sentence 1 Score: {clip_metrics['sentence1_score']:.4f} ± {clip_metrics['sentence1_std']:.4f}")
        print(f"Sentence 2 Score: {clip_metrics['sentence2_score']:.4f} ± {clip_metrics['sentence2_std']:.4f}")
        print(f"\nT-test Results (Full Prompt vs Sentence 1):")
        print(f"t-statistic: {clip_metrics['t_test_statistic']:.4f}")
        print(f"p-value: {clip_metrics['t_test_p_value']:.4f}")
        
        # Append CLIP score results to main results file
        with open(results_file, 'a', encoding='utf-8') as f:
            f.write("=== CLIP Scores ===\n")
            f.write(f"Full Prompt Score: {clip_metrics['full_prompt_score']:.4f} ± {clip_metrics['full_prompt_std']:.4f}\n")
            f.write(f"Sentence 1 Score: {clip_metrics['sentence1_score']:.4f} ± {clip_metrics['sentence1_std']:.4f}\n")
            f.write(f"Sentence 2 Score: {clip_metrics['sentence2_score']:.4f} ± {clip_metrics['sentence2_std']:.4f}\n\n")
            f.write(f"T-test Results (Full Prompt vs Sentence 1):\n")
            f.write(f"t-statistic: {clip_metrics['t_test_statistic']:.4f}\n")
            f.write(f"p-value: {clip_metrics['t_test_p_value']:.4f}\n\n")
        
        # Save overall CLIP score results
        with open(clip_results_file, 'w', encoding='utf-8') as cf:
            cf.write(f"Full Prompt Score: {clip_metrics['full_prompt_score']:.4f} ± {clip_metrics['full_prompt_std']:.4f}\n")
            cf.write(f"Sentence 1 Score: {clip_metrics['sentence1_score']:.4f} ± {clip_metrics['sentence1_std']:.4f}\n")
            cf.write(f"Sentence 2 Score: {clip_metrics['sentence2_score']:.4f} ± {clip_metrics['sentence2_std']:.4f}\n\n")
            cf.write(f"T-test Results (Full Prompt vs Sentence 1):\n")
            cf.write(f"t-statistic: {clip_metrics['t_test_statistic']:.4f}\n")
            cf.write(f"p-value: {clip_metrics['t_test_p_value']:.4f}\n")
        
        # Save per-prompt CLIP results
        with open(clip_per_prompt_file, 'w', encoding='utf-8') as cf:
            for prompt_idx in sorted(per_prompt_clip.keys()):
                metrics = per_prompt_clip[prompt_idx]
                prompt = prompt_texts[prompt_idx]
                cf.write(f"Prompt {prompt_idx + 1}:\n")
                cf.write(f"Text: {prompt}\n")
                cf.write(f"Full Prompt Score: {metrics['full_prompt_score']:.4f} ± {metrics['full_prompt_std']:.4f}\n")
                cf.write(f"Sentence 1 Score: {metrics['sentence1_score']:.4f} ± {metrics['sentence1_std']:.4f}\n")
                cf.write(f"Sentence 2 Score: {metrics['sentence2_score']:.4f} ± {metrics['sentence2_std']:.4f}\n")
                cf.write(f"T-test Results (Full Prompt vs Sentence 1):\n")
                cf.write(f"t-statistic: {metrics['t_test_statistic']:.4f}\n")
                cf.write(f"p-value: {metrics['t_test_p_value']:.4f}\n\n")
        
        print(f"CLIP score results saved to: {clip_results_file}")
        print(f"Per-prompt CLIP results saved to: {clip_per_prompt_file}")
    
    # Calculate VQA score
    if args.metric in ['vqa', 'all']:
        # Initialize VQA model
        vqa_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.vqa_model_path if args.vqa_model_path else "Qwen/Qwen2.5-VL-7B-Instruct", 
            torch_dtype="auto", device_map="auto"
        )
        vqa_processor = AutoProcessor.from_pretrained(args.vqa_model_path if args.vqa_model_path else "Qwen/Qwen2.5-VL-7B-Instruct")
        
        # Check if questions path is provided
        if args.vqa_questions_path is None:
            print("Error: VQA questions path not provided. Use --vqa_questions_path to specify a question file.")
            return
            
        # Load VQA questions from the provided file
        prompt_questions = load_vqa_questions(args.vqa_questions_path)
        
        if not prompt_questions:
            print(f"Error: No questions found in the file {args.vqa_questions_path}")
            return
            
        print(f"Loaded questions for {len(prompt_questions)} prompts from {args.vqa_questions_path}")
        
        # Calculate VQA metrics
        vqa_metrics, per_prompt_metrics = calculate_vqa_metrics(
            masked_images_path, prompt_questions, vqa_model, vqa_processor, vqa_model.device
        )
        
        # Save VQA results
        vqa_results_file = os.path.join(eval_dir, "vqa_results.json")
        with open(vqa_results_file, 'w') as f:
            json.dump({
                'overall_metrics': vqa_metrics,
                'per_prompt_metrics': per_prompt_metrics,
                'question_metrics': {
                    f'question_{i+1}': {
                        'mean': vqa_metrics.get(f'question_{i+1}_mean', 0),
                        'std': vqa_metrics.get(f'question_{i+1}_std', 0)
                    }
                    for i in range(max(len(questions) for questions in prompt_questions.values()))
                }
            }, f, indent=2)
        
        # Append VQA summary to main results file
        with open(results_file, 'a', encoding='utf-8') as f:
            f.write("=== VQA Evaluation ===\n")
            f.write(f"Overall Mean Score: {vqa_metrics['overall_mean']:.4f} ± {vqa_metrics['overall_std']:.4f}\n\n")
            f.write("Question-specific Results:\n")
            for i in range(max(len(questions) for questions in prompt_questions.values())):
                expected = "yes" if i == 0 else "no"
                f.write(f"Question {i+1} (Expected: {expected}): {vqa_metrics.get(f'question_{i+1}_mean', 0):.4f} ± {vqa_metrics.get(f'question_{i+1}_std', 0):.4f}\n")
            f.write("\nPer-Prompt Results:\n")
            for prompt_idx, metrics in sorted(per_prompt_metrics.items()):
                f.write(f"Prompt {prompt_idx + 1}:\n")
                f.write(f"Mean Score: {metrics['mean_score']:.4f} ± {metrics['std_score']:.4f}\n")
                for q_idx in range(len(prompt_questions[prompt_idx])):
                    expected = "yes" if q_idx == 0 else "no"
                    f.write(f"Question {q_idx+1} (Expected: {expected}): {np.mean(metrics['question_scores'][q_idx]):.4f} ± {np.std(metrics['question_scores'][q_idx]):.4f}\n")
                f.write("\n")
            f.write(f"Detailed results saved to: {vqa_results_file}\n\n")
            
        print(f"\nVQA results saved to: {vqa_results_file}")
        print(f"Overall Mean Score: {vqa_metrics['overall_mean']:.4f} ± {vqa_metrics['overall_std']:.4f}")
        print("\nQuestion-specific Results:")
        for i in range(max(len(questions) for questions in prompt_questions.values())):
            expected = "yes" if i == 0 else "no"
            print(f"Question {i+1} (Expected: {expected}): {vqa_metrics.get(f'question_{i+1}_mean', 0):.4f} ± {vqa_metrics.get(f'question_{i+1}_std', 0):.4f}")
    
    print(f"\nResults saved to: {results_file}")
    print(f"Per-prompt results saved to: {per_prompt_file}")
    print(f"Attribute results saved to: {attribute_results_file}")
    print(f"Per-attribute results saved to: {per_attribute_file}")

if __name__ == "__main__":
    main()