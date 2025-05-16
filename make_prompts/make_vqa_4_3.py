import re
from typing import List, Tuple, Dict, Optional

# === Constants (inspired by make_prompt_4_3.py) ===
STARTERS_LIST = ["A photo of", "A drawing of", "A painting of", "A sketch of"]
LABEL_TYPES_LIST = ["a label", "a caption", "a text", "a word"]

_starters_regex_str = '|'.join(re.escape(s) for s in STARTERS_LIST)
_label_types_regex_str = '|'.join(re.escape(l) for l in LABEL_TYPES_LIST)

# Pattern for Category 4.3 prompts:
# e.g., "A photo of 3 apples. There is a label '5 apples' written on it."
_category4_3_pattern = re.compile(
    rf"^({_starters_regex_str})\s+(\d+)\s+([\w\s]+?)\.\s*"
    rf"There is\s+({_label_types_regex_str})\s+'(\d+)\s+([\w\s]+?)'\s+written on it\.$",
    re.IGNORECASE
)

# === Helper Functions ===

def load_sentences(filepath: str) -> List[str]:
    """Loads sentences from a file, stripping whitespace and skipping empty lines."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def extract_category4_3_components(sentence: str) -> Optional[Tuple[str, str, str, str, str, str]]:
    """
    Extracts components from a category 4.3 prompt.
    Returns: (starter, n1_str, object1, label_type_str, n2_str, object2)
    Returns None if the sentence does not match the expected pattern.
    """
    match = _category4_3_pattern.fullmatch(sentence)
    if match:
        starter = match.group(1)
        n1 = match.group(2)         # e.g., "3"
        object1 = match.group(3)    # e.g., "apples"
        label_type = match.group(4) # e.g., "a label"
        n2 = match.group(5)         # e.g., "5"
        object2 = match.group(6)    # e.g., "apples"
        return starter, n1, object1.strip(), label_type, n2, object2.strip()
    return None

def generate_vqa_questions_for_category4_3(
    n1_str: str, object1: str, n2_str: str, object2: str
) -> List[str]:
    """
    Generates VQA questions for Category 4.3 prompts.
    All questions are designed to be answerable with 'yes' if the image matches the prompt.
    Args:
        n1_str: Number of main objects (e.g., "3").
        object1: Plural name of main objects (e.g., "apples").
        n2_str: Number mentioned in the label (e.g., "5").
        object2: Plural name of objects mentioned in the label (e.g., "apples").
    """
    questions = []

    # Question 1: Main object presence
    q1 = f"Does this image clearly depict {object1}?"
    questions.append(q1)

    # Question 2: Counts
    # Qwen's Locate and Count scheme. Put the answer in the form of "n1_str", which will be removed in the vqa_score.py and used for gt.
    q2 = f"Locate each {object1} in the image with bounding box descriptions and label them. After identifying all {object1}, provide the total number. Is it {n2_str}?"
    questions.append(q2)
    
    return questions

# === Main VQA Generation Function ===
def main_generate_vqa(input_filepath: str, output_filepath: str):
    """
    Loads prompts, parses them, generates VQA questions, and writes to an output file.
    """
    prompts = load_sentences(input_filepath)
    num_prompts = len(prompts)
    num_successful = 0
    num_failed = 0

    with open(output_filepath, 'w', encoding='utf-8') as f_out:
        for i, prompt_text in enumerate(prompts):
            components = extract_category4_3_components(prompt_text)
            if components:
                _, n1, obj1, _, n2, obj2 = components
                vqa_qs = generate_vqa_questions_for_category4_3(n1, obj1, n2, obj2)
                
                f_out.write(f"Prompt: {prompt_text}\n")
                for q_idx, q_text in enumerate(vqa_qs):
                    f_out.write(f"{q_text}\n")
                f_out.write("\n") # Empty line after each prompt's questions
                num_successful += 1
            else:
                print(f"Error/Warning: Could not parse prompt #{i+1}: {prompt_text}\n")
                num_failed += 1
    
    print(f"Processing complete.")
    print(f"Total prompts read: {num_prompts}")
    print(f"Successfully generated VQA for: {num_successful} prompts.")
    print(f"Failed to process: {num_failed} prompts.")

# === Execution Block ===
if __name__ == "__main__":
    DEFAULT_INPUT_PROMPT_FILE = "category4_3.txt" 
    DEFAULT_OUTPUT_VQA_FILE = "vqa_category4_3.txt"

    input_file = DEFAULT_INPUT_PROMPT_FILE
    output_file = DEFAULT_OUTPUT_VQA_FILE
    
    print(f"Starting VQA generation for Category 4.3 prompts...")
    print(f"Reading prompts from: '{input_file}'")
    print(f"Writing VQA questions to: '{output_file}'")
    
    main_generate_vqa(input_file, output_file)
    
    print(f"\n✅ VQA questions for Category 4.3 have been written to '{output_file}'") 