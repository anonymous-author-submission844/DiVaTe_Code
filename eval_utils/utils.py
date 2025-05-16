import re

def extract_text_from_prompts(prompt_path):
    """Extract text from prompts file, looking for text within single quotes.
    If multiple texts are found within quotes, they will be joined with a space.
    Also returns the number of matches for each prompt to handle special cases.
    """
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompts = f.readlines()
    
    extracted_texts = []
    visual_attributes = []
    match_counts = []  # Track the number of matches per prompt
    
    for prompt in prompts:
        # Find all text within single quotes
        matches = re.findall(r"'([^']*)'", prompt)
        if matches:
            # Join all matches with a space if there are multiple matches
            extracted_text = ' '.join(matches)
            extracted_texts.append(extracted_text)
            # Store the number of matches
            match_counts.append(len(matches))
        else:
            match_counts.append(0)

        # Find all visual attributes
        visual_attribute = prompt.split('.')[0].split('of')[-1].strip()
        if visual_attribute:
            visual_attributes.append(visual_attribute)

    return extracted_texts, visual_attributes, match_counts
