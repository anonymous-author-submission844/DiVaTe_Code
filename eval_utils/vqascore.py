import torch
from qwen_vl_utils import process_vision_info
import numpy as np
from PIL import Image
import torch.nn as nn

IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"
SYSTEM_MSG = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
IMAGE_TOKEN_INDEX = -200


def t5_tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    # Since there's no bos_token_id, simply concatenate the tokenized prompt_chunks with the image_token_index
    for x in insert_separator(prompt_chunks, [image_token_index]):
        input_ids.extend(x)

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def load_vqa_questions(path):
    """Load VQA questions from a file with specific format:
    Each prompt has multiple lines:
    - Line 1: Prompt text
    - Lines 2-N: Questions (variable number of questions)
    - Empty line to separate prompts
    
    Args:
        path: Path to the questions file
    
    Returns:
        Dictionary mapping prompt indices to their questions
    """
    prompt_questions = {}
    current_prompt_idx = 0
    current_questions = []
    is_first_line = True
    
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines at the start
        if not line and not current_questions:
            i += 1
            continue
            
        # If we hit an empty line and have questions, save the current prompt
        if not line and current_questions:
            prompt_questions[current_prompt_idx] = current_questions
            current_prompt_idx += 1
            current_questions = []
            is_first_line = True
            i += 1
            continue
            
        # Skip the first line (prompt text) of each prompt
        if is_first_line:
            is_first_line = False
            i += 1
            continue
            
        # Add question
        current_questions.append(line)
        i += 1
    
    # Don't forget to add the last prompt if there are questions
    if current_questions:
        prompt_questions[current_prompt_idx] = current_questions
                
    return prompt_questions


@torch.no_grad()
@torch.autocast(device_type='cuda', dtype=torch.float16)
def calculate_vqa_score(model, processor, image_path, questions, device):
    """Calculate VQA score for a given image and questions using Qwen2.5-VL model
    
    Args:
        model: Qwen2.5-VL model instance
        image_path: Path to the image file
        questions: List of questions to ask
        device: Device to run inference on
        
    Returns:
        Dictionary containing individual scores, questions, answers, and overall score
    """
    output_texts = []
    
    # Process each question individually
    for question in questions:
        #if "Locate" in question:
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": f"{question} Please answer yes or no."} if not "Locate" in question else {"type": "text", "text": f"{question}"},
                ],
            }
        ]
        
        # Prepare input for a single question
        text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(message)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        ).to(torch.float16)
        inputs = inputs.to(model.device)
        
        # Generate output for this question
        generated_ids = model.generate(**inputs, max_new_tokens=10 if not "Locate" in question else 256)
        generated_ids_trimmed = generated_ids[0][len(inputs.input_ids[0]):]
        output_text = processor.decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # Clean up output text
        output_text = output_text.replace('.', '').strip()
        output_text = output_text.replace('addCriterion\n', '').strip()
        output_texts.append(output_text)
        print(message[0]['content'])
        print(output_text)
    
    # Define expected answers based on question index
    # First question should be 'yes', rest should be 'no'
    expected_answers = ['yes'] + ['no'] * (len(questions) - 1)
    
    # Calculate scores for each question and overall
    scores = []
    question_scores = {i: [] for i in range(len(questions))}  # Store scores for each question type
    question_details = []  # Store detailed information for each question
    
    for i, (question, output_text, expected) in enumerate(zip(questions, output_texts, expected_answers)):
        # Check if the output contains the expected answer
        is_correct = expected in output_text.lower()
        score = 1.0 if is_correct else 0.0
        scores.append(score)
        question_scores[i].append(score)
        
        # Store detailed information for this question
        question_details.append({
            'question': question,
            'model_answer': output_text,
            'expected_answer': expected,
            'is_correct': is_correct,
            'score': score
        })
    
    # Calculate overall score (average of all scores)
    overall_score = sum(scores) / len(scores)
    
    # Return both individual scores and overall score
    return {
        'individual_scores': scores,
        'question_scores': question_scores,
        'overall_score': overall_score,
        'question_details': question_details
    }

def calculate_vqa_metrics(masked_images_path, prompt_questions, model, processor, device):
    """Calculate VQA metrics for all images, organized by prompt index.
    
    Args:
        masked_images_path: Path to directory containing masked images
        prompt_questions: Dictionary mapping prompt indices to their questions
        model: VQA model
        
    Returns:
        Dictionary containing overall metrics and per-prompt metrics
    """
    import os
    from tqdm import tqdm
    
    # Dictionary to store metrics for each prompt
    prompt_metrics = {}
    total_scores = []
    
    # Get list of image files
    image_files = [f for f in sorted(os.listdir(masked_images_path)) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Process images with progress bar
    for image_name in tqdm(image_files, desc="Calculating VQA scores"):
        # Get the prompt index from the image name
        prompt_idx = int(image_name.split('_')[2]) - 1
        
        # Skip if we don't have questions for this prompt
        if prompt_idx not in prompt_questions:
            continue
            
        # Initialize prompt metrics if not exists
        if prompt_idx not in prompt_metrics:
            prompt_metrics[prompt_idx] = {
                'scores': [],
                'mean_score': 0.0,
                'std_score': 0.0,
                'question_scores': {i: [] for i in range(len(prompt_questions[prompt_idx]))},  # Store scores for each question separately
                'question_details': []  # Store detailed information for each question
            }
            
        # Load image and calculate scores
        image_path = os.path.join(masked_images_path, image_name)
        scores = calculate_vqa_score(model, processor, image_path, prompt_questions[prompt_idx], model.device)
        
        # Store scores for each question separately
        for q_idx, score in enumerate(scores['individual_scores']):
            prompt_metrics[prompt_idx]['question_scores'][q_idx].append(score)
        
        # Store overall scores
        prompt_metrics[prompt_idx]['scores'].append(scores['individual_scores'])
        total_scores.extend(scores['individual_scores'])
        
        # Store detailed question information
        prompt_metrics[prompt_idx]['question_details'].append({
            'image_name': image_name,
            'questions': scores['question_details']
        })
    
    # Calculate final metrics
    final_metrics = {
        'overall_mean': np.mean(total_scores),
        'overall_std': np.std(total_scores)
    }
    
    # Calculate per-question metrics
    max_questions = max(len(questions) for questions in prompt_questions.values())
    for q_idx in range(max_questions):
        question_scores = []
        for metrics in prompt_metrics.values():
            if q_idx in metrics['question_scores']:
                question_scores.extend(metrics['question_scores'][q_idx])
        
        if question_scores:  # Only add metrics if we have scores for this question
            final_metrics[f'question_{q_idx+1}_mean'] = np.mean(question_scores)
            final_metrics[f'question_{q_idx+1}_std'] = np.std(question_scores)
    
    return final_metrics, prompt_metrics