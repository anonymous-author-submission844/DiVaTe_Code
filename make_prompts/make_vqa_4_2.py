import re
from typing import List, Tuple, Dict

# === Constants (inspired by make_vqa_4_1.py and make_prompt_4_2.py) ===
STARTERS_LIST = ["A photo of", "A drawing of", "A painting of", "A sketch of"]
LABEL_TYPES_LIST = ["a label", "a caption", "a text", "a word"]

HEADING_DIRECTIONS_PAIRS: List[Tuple[str, str]] = [("east", "west"), ("west", "east")]
POSITIONAL_DIRECTIONS_PAIRS: List[Tuple[str, str]] = [
    ("front", "back"), ("back", "front"),
    ("left", "right"), ("right", "left"),
    ("up", "down"), ("down", "up"),
    ("top", "bottom"), ("bottom", "top")
]

ALL_DIRECTIONS_PAIRS = HEADING_DIRECTIONS_PAIRS + POSITIONAL_DIRECTIONS_PAIRS
OPPOSITE_DIRECTIONS_MAP: Dict[str, str] = {d1: d2 for d1, d2 in ALL_DIRECTIONS_PAIRS}
OPPOSITE_DIRECTIONS_MAP.update({d2: d1 for d1, d2 in ALL_DIRECTIONS_PAIRS})


# Regex patterns for parsing prompts
_starters_regex_str = '|'.join(re.escape(s) for s in STARTERS_LIST)
_label_types_regex_str = '|'.join(re.escape(l) for l in LABEL_TYPES_LIST)

# Individual directions for regex matching
_heading_dirs_individual_regex_str = '|'.join(re.escape(d) for d in set(sum(HEADING_DIRECTIONS_PAIRS, ())))
_pos_dirs_individual_regex_str = '|'.join(re.escape(d) for d in set(sum(POSITIONAL_DIRECTIONS_PAIRS, ())))


# Pattern for Heading Case:
# e.g., "A photo of two beavers, one beaver is heading east and the other is heading west.
#        The one heading east has a label 'heading west', and the one heading west has a label 'heading east' on it."
_heading_pattern = re.compile(
    rf"({_starters_regex_str}) two ([\w-]+?)s, "  # G1: starter, G2: singular_obj (non-greedy, includes hyphens)
    rf"one \2 is heading ({_heading_dirs_individual_regex_str}) and the other is heading ({_heading_dirs_individual_regex_str})\. "  # G3: dir1, G4: dir2
    rf"The one heading \3 has ({_label_types_regex_str}) 'heading \4', "  # G5: label_type
    rf"and the one heading \4 has \5 'heading \3' on it\."
)

# Pattern for Positional Case:
# e.g., "A photo of two roses, one rose is on the left and the other is on the right.
#        The one on the left has a label 'right', and the one on the right has a label 'left' on it."
_positional_pattern = re.compile(
    rf"({_starters_regex_str}) two ([\w-]+?)s, "  # G1: starter, G2: singular_obj (non-greedy, includes hyphens)
    rf"one \2 is on the ({_pos_dirs_individual_regex_str}) and the other is on the ({_pos_dirs_individual_regex_str})\. "  # G3: pos1, G4: pos2
    rf"The one on the \3 has ({_label_types_regex_str}) '\4', "  # G5: label_type
    rf"and the one on the \4 has \5 '\3' on it\."
)

# === Helper Functions ===

def load_sentences(filepath: str) -> List[str]:
    """Loads sentences from a file, stripping whitespace and skipping empty lines."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def extract_category4_2_components(sentence: str) -> Tuple[str, str, str, str, str, str]:
    """
    Extracts components from a category 4.2 prompt.
    Returns: (case_type, starter, singular_obj, loc1, loc2, label_type)
    'loc1' and 'loc2' are the actual directions/positions.
    """
    heading_match = _heading_pattern.fullmatch(sentence)
    if heading_match:
        starter = heading_match.group(1)
        singular_obj = heading_match.group(2)
        loc1 = heading_match.group(3) # Actual direction of first object
        loc2 = heading_match.group(4) # Actual direction of second object
        label_type = heading_match.group(5)

        if OPPOSITE_DIRECTIONS_MAP.get(loc1) != loc2:
            raise ValueError(
                f"Inconsistent heading directions in prompt: "
                f"Direction '{loc1}' and '{loc2}' are not opposites. Sentence: {sentence}"
            )
        return "heading", starter, singular_obj, loc1, loc2, label_type

    positional_match = _positional_pattern.fullmatch(sentence)
    if positional_match:
        starter = positional_match.group(1)
        singular_obj = positional_match.group(2)
        loc1 = positional_match.group(3) # Actual position of first object
        loc2 = positional_match.group(4) # Actual position of second object
        label_type = positional_match.group(5)

        if OPPOSITE_DIRECTIONS_MAP.get(loc1) != loc2:
            raise ValueError(
                f"Inconsistent positional directions in prompt: "
                f"Position '{loc1}' and '{loc2}' are not opposites. Sentence: {sentence}"
            )
        return "positional", starter, singular_obj, loc1, loc2, label_type

    raise ValueError(f"Failed to parse sentence for Category 4.2 (no pattern matched): {sentence}")


def generate_vqa_questions_for_category4_2(case_type: str, singular_obj: str, loc1: str, loc2: str) -> List[str]:
    """
    Generates VQA questions based on parsed components from Category 4.2 prompts.
    - case_type: 'heading' or 'positional'.
    - singular_obj: The singular form of the object (e.g., "beaver", "rose").
    - loc1: The direction/position of the first object (e.g., "east", "left").
    - loc2: The direction/position of the second object (e.g., "west", "right").
    """
    # make_prompt_4_2.py uses simple "{obj}s" for plural, so we replicate that here.
    plural_obj_form = f"{singular_obj}s"
    
    new_questions = []

    # Question 1: Verify presence of two distinct objects
    q1 = f"Does this image clearly depict two distinct {plural_obj_form}?"
    new_questions.append(q1)

    # Question 2: Verify the spatial relationship
    if case_type == "heading":
        q2 = f"Does the image incorrectly depict the spatial arrangement where one {singular_obj} is heading {loc1} and the other {singular_obj} is heading {loc2}?"
        new_questions.append(q2)
    elif case_type == "positional":
        q2 = f"Does the image incorrectly depict the spatial arrangement where one {singular_obj} is on the {loc1} and the other {singular_obj} is on the {loc2}?"
        new_questions.append(q2)
    else:
        # This case should ideally not be reached if extract_category4_2_components is robust.
        print(f"Warning: Unknown case_type encountered for VQA generation: {case_type}")
        # q2 = f"Could not determine appropriate spatial question for: {plural_obj_form} with locations {loc1}, {loc2}"
        # new_questions.append(q2) # Avoid adding a confusing default question

    return new_questions

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
            try:
                case, _, s_obj, location1, location2, _ = extract_category4_2_components(prompt_text)
                vqa_qs = generate_vqa_questions_for_category4_2(case, s_obj, location1, location2)
                
                f_out.write(f"Prompt: {prompt_text}\n")
                for q_idx, q_text in enumerate(vqa_qs):
                    f_out.write(f"{q_text}\n")
                f_out.write("\n") # Extra newline between prompt blocks
                num_successful += 1
            except ValueError as e:
                print(f"Error processing prompt #{i+1}: {prompt_text}\n -> {e}\n")
                num_failed += 1
    
    print(f"Processing complete.")
    print(f"Total prompts read: {num_prompts}")
    print(f"Successfully generated VQA for: {num_successful} prompts.")
    print(f"Failed to process: {num_failed} prompts.")

# === Execution Block ===
if __name__ == "__main__":
    # Default file paths. Assumes 'make_prompt_4_2.py' saves to 'category4_2_spatial.txt'
    # in the same directory as this script, or in the current working directory if run from elsewhere.
    DEFAULT_INPUT_PROMPT_FILE = "category4_2_spatial.txt" 
    DEFAULT_OUTPUT_VQA_FILE = "vqa_category4_2.txt"

    # You can modify these to use different file paths if needed.
    # For example, by uncommenting and setting command-line argument parsing.
    # import argparse
    # parser = argparse.ArgumentParser(description="Generate VQA questions for Category 4.2 prompts.")
    # parser.add_argument("--input_file", type=str, default=DEFAULT_INPUT_PROMPT_FILE,
    #                     help="Path to the input file containing Category 4.2 prompts.")
    # parser.add_argument("--output_file", type=str, default=DEFAULT_OUTPUT_VQA_FILE,
    #                     help="Path to the output file where VQA questions will be written.")
    # args = parser.parse_args()
    # input_file = args.input_file
    # output_file = args.output_file
    
    input_file = DEFAULT_INPUT_PROMPT_FILE
    output_file = DEFAULT_OUTPUT_VQA_FILE
    
    print(f"Starting VQA generation for Category 4.2 prompts...")
    print(f"Reading prompts from: '{input_file}'")
    print(f"Writing VQA questions to: '{output_file}'")
    
    main_generate_vqa(input_file, output_file)
    
    print(f"✅ VQA questions for Category 4.2 have been written to '{output_file}'") 