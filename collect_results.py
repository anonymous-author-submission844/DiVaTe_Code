import os
import pandas as pd
import glob
import re
import argparse
import json

def extract_metrics_from_file(file_path):
    """Extract metrics from a single evaluation results file."""
    metrics = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extract accuracy and edit distance using regex
            accuracy_match = re.search(r'Accuracy:\s*([\d.]+)', content)
            edit_dist_match = re.search(r'Edit Distance:\s*([\d.]+)', content)
            
            if accuracy_match:
                metrics['accuracy'] = float(accuracy_match.group(1))
            if edit_dist_match:
                metrics['edit_distance'] = float(edit_dist_match.group(1))
                
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    
    return metrics

def extract_clip_scores_from_file(file_path):
    """Extract CLIP scores from a single evaluation results file."""
    metrics = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extract sentence scores and standard deviations using regex
            full_prompt_match = re.search(r'Full Prompt Score:\s*([\d.]+)\s*±\s*([\d.]+)', content)
            sentence1_match = re.search(r'Sentence 1 Score:\s*([\d.]+)\s*±\s*([\d.]+)', content)
            sentence2_match = re.search(r'Sentence 2 Score:\s*([\d.]+)\s*±\s*([\d.]+)', content)
            
            # Extract t-test results
            t_stat_match = re.search(r't-statistic:\s*([\d.-]+)', content)
            p_value_match = re.search(r'p-value:\s*([\d.]+)', content)
            
            if full_prompt_match:
                metrics['full_prompt_score'] = float(full_prompt_match.group(1))
                metrics['full_prompt_std'] = float(full_prompt_match.group(2))
            if sentence1_match:
                metrics['sentence1_score'] = float(sentence1_match.group(1))
                metrics['sentence1_std'] = float(sentence1_match.group(2))
            if sentence2_match:
                metrics['sentence2_score'] = float(sentence2_match.group(1))
                metrics['sentence2_std'] = float(sentence2_match.group(2))
            if t_stat_match:
                metrics['t_test_statistic'] = float(t_stat_match.group(1))
            if p_value_match:
                metrics['t_test_p_value'] = float(p_value_match.group(1))
                
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    
    return metrics

def extract_vqa_scores_from_file(file_path):
    """Extract VQA scores from a single evaluation results file."""
    metrics = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Extract overall metrics
            overall_metrics = data.get('overall_metrics', {})
            metrics['vqa_overall_mean'] = overall_metrics.get('overall_mean')
            metrics['vqa_overall_std'] = overall_metrics.get('overall_std')
            
            # Extract per-question metrics
            for key, value in overall_metrics.items():
                if key.startswith('question_'):
                    metrics[f'vqa_{key}'] = value
                    
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    
    return metrics

def collect_results(base_dir, model, categories):
    """Collect results from all categories for a single model."""
    results = []
    
    for category in categories:
        # Construct the path to the evaluation results
        eval_dir = os.path.join(base_dir, model, category, "evaluation_results")
        
        # Initialize metrics for this category
        category_metrics = {
            'model': model,
            'category': category,
            'prompt_accuracy': None,
            'prompt_edit_distance': None,
            'attribute_accuracy': None,
            'attribute_edit_distance': None,
            'clip_full_prompt_score': None,
            'clip_full_prompt_std': None,
            'clip_sentence1_score': None,
            'clip_sentence1_std': None,
            'clip_sentence2_score': None,
            'clip_sentence2_std': None,
            'clip_t_test_statistic': None,
            'clip_t_test_p_value': None,
            'vqa_overall_mean': None,
            'vqa_overall_std': None
        }
        
        # Process prompt text results
        prompt_results_file = os.path.join(eval_dir, "evaluation_results.txt")
        if os.path.exists(prompt_results_file):
            metrics = extract_metrics_from_file(prompt_results_file)
            if metrics:
                category_metrics['prompt_accuracy'] = metrics.get('accuracy')
                category_metrics['prompt_edit_distance'] = metrics.get('edit_distance')
        else:
            print(f"Warning: Prompt results file not found: {prompt_results_file}")
        
        # Process attribute results
        attr_results_file = os.path.join(eval_dir, "attribute_evaluation_results.txt")
        if os.path.exists(attr_results_file):
            metrics = extract_metrics_from_file(attr_results_file)
            if metrics:
                category_metrics['attribute_accuracy'] = metrics.get('accuracy')
                category_metrics['attribute_edit_distance'] = metrics.get('edit_distance')
        else:
            print(f"Warning: Attribute results file not found: {attr_results_file}")
            
        # Process CLIP score results
        clip_results_file = os.path.join(eval_dir, "clip_evaluation_results.txt")
        if os.path.exists(clip_results_file):
            metrics = extract_clip_scores_from_file(clip_results_file)
            if metrics:
                category_metrics['clip_full_prompt_score'] = metrics.get('full_prompt_score')
                category_metrics['clip_full_prompt_std'] = metrics.get('full_prompt_std')
                category_metrics['clip_sentence1_score'] = metrics.get('sentence1_score')
                category_metrics['clip_sentence1_std'] = metrics.get('sentence1_std')
                category_metrics['clip_sentence2_score'] = metrics.get('sentence2_score')
                category_metrics['clip_sentence2_std'] = metrics.get('sentence2_std')
                category_metrics['clip_t_test_statistic'] = metrics.get('t_test_statistic')
                category_metrics['clip_t_test_p_value'] = metrics.get('t_test_p_value')
        else:
            print(f"Warning: CLIP results file not found: {clip_results_file}")
            
        # Process VQA results
        vqa_results_file = os.path.join(eval_dir, "vqa_results.json")
        if os.path.exists(vqa_results_file):
            metrics = extract_vqa_scores_from_file(vqa_results_file)
            if metrics:
                # Update category metrics with VQA scores
                category_metrics.update(metrics)
        else:
            print(f"Warning: VQA results file not found: {vqa_results_file}")
        
        results.append(category_metrics)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Collect evaluation results for a single model across all categories")
    parser.add_argument("--model", type=str, required=True, help="Model name to process (e.g., sana)")
    parser.add_argument("--base_dir", type=str, default="<your_dir>",
                      help="Base directory containing model results")
    args = parser.parse_args()
    
    # List of categories to process
    categories = [
        "category1",
        "category2_1", "category2_2", "category2_3",
        "category3_1", "category3_2", "category3_3", "category3_4",
        "category3_5", "category3_6", "category3_7",
        "category4_1", "category4_2", "category4_3"
    ]
    
    # Collect all results
    results = collect_results(args.base_dir, args.model, categories)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Get all VQA question columns
    vqa_columns = [col for col in df.columns if col.startswith('vqa_')]
    
    # Reorder columns
    df = df[['model', 'category', 
             'prompt_accuracy', 'prompt_edit_distance',
             'attribute_accuracy', 'attribute_edit_distance',
             'clip_full_prompt_score', 'clip_full_prompt_std',
             'clip_sentence1_score', 'clip_sentence1_std',
             'clip_sentence2_score', 'clip_sentence2_std',
             'clip_t_test_statistic', 'clip_t_test_p_value'] + vqa_columns]
    
    # Save to CSV in the model's directory
    output_file = os.path.join(args.base_dir, args.model, "evaluation_results_summary.csv")
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    summary = df[['prompt_accuracy', 'prompt_edit_distance',
                 'attribute_accuracy', 'attribute_edit_distance',
                 'clip_full_prompt_score', 'clip_full_prompt_std',
                 'clip_sentence1_score', 'clip_sentence1_std', 
                 'clip_sentence2_score', 'clip_sentence2_std',
                 'clip_t_test_statistic', 'clip_t_test_p_value'] + vqa_columns].mean()
    print(summary)
    
    print("\nSummary Statistics by Category:")
    summary = df.groupby('category')[['prompt_accuracy', 'prompt_edit_distance',
                                    'attribute_accuracy', 'attribute_edit_distance',
                                    'clip_full_prompt_score', 'clip_full_prompt_std',
                                    'clip_sentence1_score', 'clip_sentence1_std',
                                    'clip_sentence2_score', 'clip_sentence2_std',
                                    'clip_t_test_statistic', 'clip_t_test_p_value'] + vqa_columns].mean()
    print(summary)
    
    # Print t-test summary
    print("\nT-test Summary:")
    t_test_summary = df[['category', 'clip_t_test_statistic', 'clip_t_test_p_value']]
    print(t_test_summary)

if __name__ == "__main__":
    main() 