import json
from collections import Counter, defaultdict
import textdistance

def calculate_ocr_metrics(ocr_results_path, target_items, match_counts=None):
    """Calculate OCR metrics including accuracy and normalized edit distance.
    Returns both overall metrics and per-item metrics.
    For prompts with exactly two quoted texts, both orderings are tried and the better result is selected.
    
    Args:
        ocr_results_path: Path to JSON file containing OCR results
        target_items: List of target items to detect (either prompt texts or visual attributes)
        match_counts: List indicating the number of matches found in each prompt
    """
    with open(ocr_results_path, 'r', encoding='utf-8') as f:
        ocr_data = json.load(f)
    
    # Dictionary to store metrics for each item
    item_metrics = defaultdict(lambda: {
        'accuracy': 0,
        'edit_distance': 0,
        'count': 0
    })
    
    for image_name, data in sorted(ocr_data.items()):
        try:
            # Extract prompt_idx from filename (prompt_XXXX_Y.png)
            parts = image_name.split('_')
            prompt_idx = int(parts[1]) - 1
            if prompt_idx >= len(target_items):
                continue
                
            # Get target item and convert to lowercase
            target_item = target_items[prompt_idx].lower()
            
            # Get OCR results for this image
            ocr_results = data.get('ocr_results', [])
            
            # If OCR results are empty, treat as zero scores
            if not ocr_results:
                item_metrics[prompt_idx]['accuracy'] += 0
                item_metrics[prompt_idx]['edit_distance'] += 1.0
                item_metrics[prompt_idx]['count'] += 1
                continue
            
            # Get all OCR results for this image
            ocr_text = ' '.join([result['text'].lower() for result in ocr_results])
            
            # Check if this prompt had exactly two matches and try both orderings
            if match_counts and prompt_idx < len(match_counts) and match_counts[prompt_idx] == 2:
                # Split the text and try both orderings
                target_words = target_item.split()
                if len(target_words) == 2:  # Safety check
                    normal_order = ' '.join(target_words)
                    reversed_order = ' '.join([target_words[1], target_words[0]])
                    
                    # Calculate metrics for both orderings
                    normal_accuracy = 1 if normal_order in ocr_text else 0
                    reversed_accuracy = 1 if reversed_order in ocr_text else 0
                    
                    normal_distance = textdistance.levenshtein.normalized_distance(normal_order, ocr_text)
                    reversed_distance = textdistance.levenshtein.normalized_distance(reversed_order, ocr_text)
                    
                    # Select the better result (higher accuracy or lower edit distance if accuracies are equal)
                    if normal_accuracy > reversed_accuracy or (normal_accuracy == reversed_accuracy and normal_distance < reversed_distance):
                        accuracy = normal_accuracy
                        normalized_distance = normal_distance
                    else:
                        accuracy = reversed_accuracy
                        normalized_distance = reversed_distance
                else:
                    # If not exactly two words, use regular matching
                    accuracy = 1 if target_item in ocr_text else 0
                    normalized_distance = textdistance.levenshtein.normalized_distance(target_item, ocr_text)
            else:
                # For other cases, just calculate metrics directly
                accuracy = 1 if target_item in ocr_text else 0
                normalized_distance = textdistance.levenshtein.normalized_distance(target_item, ocr_text)
            
            # Add to item metrics using prompt_idx as key
            item_metrics[prompt_idx]['accuracy'] += accuracy
            item_metrics[prompt_idx]['edit_distance'] += normalized_distance
            item_metrics[prompt_idx]['count'] += 1
            
        except (IndexError, ValueError):
            continue
    
    # Calculate final metrics by averaging across items
    final_metrics = {
        'accuracy': 0,
        'edit_distance': 0
    }
    
    # Calculate per-item metrics
    per_item_metrics = {}
    valid_items = 0
    
    for prompt_idx, metrics in item_metrics.items():
        # Average the metrics for this item's images
        per_item_metrics[prompt_idx] = {
            'accuracy': metrics['accuracy'] / metrics['count'],
            'edit_distance': metrics['edit_distance'] / metrics['count']
        }
        
        # Add to final metrics
        final_metrics['accuracy'] += per_item_metrics[prompt_idx]['accuracy']
        final_metrics['edit_distance'] += per_item_metrics[prompt_idx]['edit_distance']
        valid_items += 1
    
    # Final average across all items
    if valid_items > 0:
        final_metrics['accuracy'] /= valid_items
        final_metrics['edit_distance'] /= valid_items
    
    return final_metrics, per_item_metrics