import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import numpy as np
from scipy import stats

def calculate_clip_score(masked_images_path, prompt_path, clip_model, clip_processor):
    """Calculate CLIP score between masked images and prompt sentences.
    Returns overall metrics and per-prompt metrics."""
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompts = f.readlines()
    
    # Dictionary to store metrics for each prompt
    prompt_metrics = {}
    total_scores = {
        'sentence1': [],
        'sentence2': [],
        'full_prompt': []
    }
    
    # Get list of image files
    image_files = [f for f in sorted(os.listdir(masked_images_path)) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Process images with progress bar
    for image_name in tqdm(image_files, desc="Calculating CLIP scores"):
        try:
            # Get the prompt index from the image name
            prompt_idx = int(image_name.split('_')[2]) - 1
            if prompt_idx >= len(prompts):
                continue
                
            # Initialize prompt metrics if not exists
            if prompt_idx not in prompt_metrics:
                prompt_metrics[prompt_idx] = {
                    'sentence1_scores': [],
                    'sentence2_scores': [],
                    'full_prompt_scores': []
                }
                
            # Load masked image
            image_path = os.path.join(masked_images_path, image_name)
            image = Image.open(image_path)
            
            # Get the prompt and split into sentences
            prompt = prompts[prompt_idx].strip()
            sentences = [s.strip() for s in prompt.split('.') if s.strip()]
            if len(sentences) < 2:
                continue
                
            # Process image and all text inputs at once
            image_inputs = clip_processor(images=image, return_tensors="pt", padding=True)
            text_inputs = clip_processor(text=[prompt] + sentences, return_tensors="pt", padding=True)

            # Move to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                image_features = clip_model.get_image_features(**image_inputs)
                text_features = clip_model.get_text_features(**text_inputs)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                
                # Calculate similarity for all text inputs at once
                similarity = (image_features @ text_features.T).squeeze()
                
                # Store scores
                prompt_metrics[prompt_idx]['full_prompt_scores'].append(similarity[0].item())
                prompt_metrics[prompt_idx]['sentence1_scores'].append(similarity[1].item())
                prompt_metrics[prompt_idx]['sentence2_scores'].append(similarity[2].item())
                
                # Add to total scores
                total_scores['full_prompt'].append(similarity[0].item())
                total_scores['sentence1'].append(similarity[1].item())
                total_scores['sentence2'].append(similarity[2].item())
                
        except Exception as e:
            print(f"Error processing {image_name}: {str(e)}")
            continue
    
    # Calculate final metrics with standard deviation
    final_metrics = {
        'sentence1_score': np.mean(total_scores['sentence1']),
        'sentence1_std': np.std(total_scores['sentence1']),
        'sentence2_score': np.mean(total_scores['sentence2']),
        'sentence2_std': np.std(total_scores['sentence2']),
        'full_prompt_score': np.mean(total_scores['full_prompt']),
        'full_prompt_std': np.std(total_scores['full_prompt'])
    }
    
    # Perform t-test between full prompt and sentence1 scores
    t_stat, p_value = stats.ttest_rel(total_scores['full_prompt'], total_scores['sentence1'])
    final_metrics['t_test_statistic'] = t_stat
    final_metrics['t_test_p_value'] = p_value
    
    # Calculate per-prompt metrics with standard deviation
    per_prompt_metrics = {}
    for prompt_idx, metrics in prompt_metrics.items():
        per_prompt_metrics[prompt_idx] = {
            'sentence1_score': np.mean(metrics['sentence1_scores']),
            'sentence1_std': np.std(metrics['sentence1_scores']),
            'sentence2_score': np.mean(metrics['sentence2_scores']),
            'sentence2_std': np.std(metrics['sentence2_scores']),
            'full_prompt_score': np.mean(metrics['full_prompt_scores']),
            'full_prompt_std': np.std(metrics['full_prompt_scores'])
        }
        
        # Perform t-test for each prompt
        t_stat, p_value = stats.ttest_rel(metrics['full_prompt_scores'], metrics['sentence1_scores'])
        per_prompt_metrics[prompt_idx]['t_test_statistic'] = t_stat
        per_prompt_metrics[prompt_idx]['t_test_p_value'] = p_value
    
    return final_metrics, per_prompt_metrics