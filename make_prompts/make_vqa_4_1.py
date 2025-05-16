import re
from typing import List, Tuple, Dict

# === Constants (adapted from make_prompt_4_1.py and inspired by make_vqa_2_2.py) ===
STARTERS_LIST = ["A photo of", "A drawing of", "A painting of", "A sketch of"]
LABEL_TYPES_LIST = ["a label", "a caption", "a text", "a word"]

# Object names from make_prompt_4_1.py for accurate singularization
_OBJECT_SINGULAR_FORMS = [
    "bottle", "glass", "cup", "box", "stone", "wooden ball", "candle", "ruler"
]

# Helper function from make_prompt_4_1.py to generate plurals consistently
def _local_pluralize(word: str) -> str:
    if word == "glass": return "glasses"
    # Simplified: category4_1 objects don't end in 'y' like 'category'
    # elif word.endswith("y") and not word.endswith("ey"): return word[:-1] + "ies"
    if word.endswith(("s", "x", "ch", "sh")): return word + "es"
    return word + "s"

PLURAL_TO_SINGULAR_MAP: Dict[str, str] = {
    _local_pluralize(s): s for s in _OBJECT_SINGULAR_FORMS
}

COMPARATIVE_TO_LESS_MAP: Dict[str, str] = {
    "heavier": "lighter",
    "longer": "shorter",
    "larger": "smaller"
}

# Regex patterns for parsing prompts
_starters_regex_str = '|'.join(re.escape(s) for s in STARTERS_LIST)
_label_types_regex_str = '|'.join(re.escape(l) for l in LABEL_TYPES_LIST)

# Pattern for Non-Identical Case:
# e.g., "A photo of two bottles next to each other, where the left one is larger than the right one. The larger one has a label '10L', and the smaller one has a label '20L' on it."
_non_ident_pattern = re.compile(
    rf"({_starters_regex_str}) two ([\w\s]+?) next to each other, "  # G1: starter, G2: plural_obj
    rf"where the left one is (heavier|longer|larger) than the right one\. "  # G3: attr_type
    rf"The \3 one has ({_label_types_regex_str}) '([^']*)', "  # G3 (ref), G4: label_type, G5: label1
    rf"and the (lighter|shorter|smaller) one has \4 '([^']*)' on it\."  # G6: less_term, G4 (ref to label_type), G7: label2
)

# === Helper Functions ===

def load_sentences(filepath: str) -> List[str]:
    """Loads sentences from a file, stripping whitespace and skipping empty lines."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def get_singular(plural_form: str) -> str:
    """Converts a plural object form to its singular form using a predefined map."""
    singular = PLURAL_TO_SINGULAR_MAP.get(plural_form)
    if singular:
        return singular
    # Basic fallback if not in map (should ideally always be in map for expected inputs)
    if plural_form.endswith("s") and plural_form != "glasses": # "glasses" is special
        return plural_form[:-1]
    return plural_form


def extract_category4_1_components(sentence: str) -> Tuple[str, str, str, str, str, str, str]:
    """
    Extracts components from a category 4.1 prompt.
    First tries to parse as an 'identical' case using keyword detection and simpler regexes.
    If that fails or keywords are not present, tries the 'non-identical' pattern.
    Returns: (case_type, starter, plural_obj, main_attribute, label_type, label1, label2)
    'main_attribute' is 'noun_attr' (e.g., weight) for identical cases,
                     or 'attr_type' (e.g., heavier) for non-identical cases.
    """
    # Try simplified "identical" case detection and extraction first
    # Expected structure: "{starter} two identical {plural_obj} next to each other, which have same {noun_attr}. The left one has {label_type} '{label1}', and the right one has {label_type} '{label2}' on it."
    if " identical " in sentence and " same " in sentence:
        try:
            # Extract starter
            starter_match = re.match(rf"^({_starters_regex_str})", sentence)
            if not starter_match: raise ValueError("Starter not found in supposed identical sentence")
            starter = starter_match.group(1)

            # Extract plural object
            plural_obj_match = re.search(r"two identical ([\w\s]+?) next to each other", sentence)
            if not plural_obj_match: raise ValueError("Plural object not found in supposed identical sentence")
            plural_obj = plural_obj_match.group(1).strip()

            # Extract noun attribute
            noun_attr_match = re.search(r"have same (weight|length|size)\.", sentence)
            if not noun_attr_match: raise ValueError("Noun attribute (weight/length/size) not found in supposed identical sentence")
            noun_attr = noun_attr_match.group(1)

            # Extract label type and labels (this part is quite specific)
            # Uses \1 to ensure the label type is the same for both
            label_details_match = re.search(
                rf"The left one has ({_label_types_regex_str}) '([^\']*)', and the right one has \1 '([^\']*)' on it.",
                sentence
            )
            if not label_details_match: raise ValueError("Label details (type, label1, label2) not found or mismatched in supposed identical sentence")
            label_type = label_details_match.group(1)
            label1 = label_details_match.group(2)
            label2 = label_details_match.group(3)

            return "identical", starter, plural_obj, noun_attr, label_type, label1, label2
        except ValueError as e: # Catch specific ValueErrors from extraction steps
            # This means keywords were present, but the detailed structure didn't match expectations.
            # We can choose to fall through to non-identical or raise a more specific error.
            # For now, if keywords suggest "identical" but structure fails, it implies a malformed "identical" prompt.
            raise ValueError(f"Malformed 'identical' prompt structure despite keywords: {sentence}. Details: {e}")

    # If not parsed as "identical" by keywords, try the "non-identical" pattern
    non_ident_match = _non_ident_pattern.fullmatch(sentence)
    if non_ident_match:
        starter = non_ident_match.group(1)
        plural_obj = non_ident_match.group(2).strip()
        attr_type = non_ident_match.group(3)  # e.g., "heavier", "longer", "larger"
        
        # Corrected group indexing:
        label_type = non_ident_match.group(4) # Was non_ident_match.group(5)
        label1 = non_ident_match.group(5)     # Was non_ident_match.group(6)
        less_term_captured = non_ident_match.group(6) # Was non_ident_match.group(7)
        label2 = non_ident_match.group(7)     # Was non_ident_match.group(8)

        expected_less_term = COMPARATIVE_TO_LESS_MAP.get(attr_type)
        if expected_less_term != less_term_captured:
            raise ValueError(
                f"Inconsistent attributes in non-identical prompt: "
                f"Attribute '{attr_type}' implies less_term '{expected_less_term}', but got '{less_term_captured}'. "
                f"Sentence: {sentence}"
            )
        return "non-identical", starter, plural_obj, attr_type, label_type, label1, label2

    raise ValueError(f"Failed to parse sentence for Category 4.1 (no pattern matched): {sentence}")


def generate_vqa_questions_for_category4_1(case_type: str, plural_obj: str, main_attribute: str) -> List[str]:
    """
    Generates VQA questions based on parsed components from Category 4.1 prompts.
    - case_type: 'identical' or 'non-identical'.
    - plural_obj: The plural form of the object (e.g., "bottles").
    - main_attribute:
        For 'identical' cases, this is the noun form of the attribute (e.g., "weight", "size").
        For 'non-identical' cases, this is the comparative adjective (e.g., "heavier", "larger").
    """
    new_questions = []

    # New first question about the two objects from the first sentence
    # The "first sentence" is the prompt itself in this context, and p_obj refers to the two objects.
    new_first_q = f"Does this image clearly depict two distinct {plural_obj}?"
    new_questions.append(new_first_q)

    # Existing questions
    if case_type == "identical":
        # User request: "두개가 identical한지"
        # Prompt implies: "...have same {noun_attr}."
        q = f"Do the two {plural_obj} in the image show a noticeable visual difference in size, when they are supposed to be identical or similar?"
        #f"Are the two {plural_obj} shown in the image identical, especially in their {main_attribute}?" # main_attribute is noun_attr here
        new_questions.append(q)
    elif case_type == "non-identical":
        # User request: "실제로 왼쪽게 더 큰 지로 질문" (Is the left one actually {attr_type}?)
        # Prompt implies: "...left one is {attr_type} than the right one."
        singular_obj = get_singular(plural_obj)
        q = f"Does the image incorrectly depict the relationship where the left {singular_obj} should be {main_attribute} than the right {singular_obj}?"
        # f"According to the image, is the left {singular_obj} {main_attribute} than the right one?" # main_attribute is attr_type here
        new_questions.append(q)
    else:
        # This case should ideally not be reached if extract_category4_1_components is robust.
        print(f"Warning: Unknown case_type encountered: {case_type}")
        # Avoid adding a confusing default question if case_type is unknown,
        # the new_first_q will still be there.
        # new_questions.append(f"Could not determine appropriate question for: {plural_obj} with attribute {main_attribute}")
        
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
                case, _, p_obj, m_attr, _, _, _ = extract_category4_1_components(prompt_text)
                vqa_qs = generate_vqa_questions_for_category4_1(case, p_obj, m_attr)
                
                f_out.write(f"Prompt: {prompt_text}\n")
                for q_idx, q_text in enumerate(vqa_qs):
                    f_out.write(f"{q_text}\n")
                f_out.write("\n")
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
    # Default file paths, assuming 'make_prompt_4_1.py' saves to 'category4_1_quantity.txt'
    # OR the user provides 'category4_1.txt' from another source.
    # Adjust if your files are located elsewhere.
    DEFAULT_INPUT_PROMPT_FILE = "category4_1.txt" # Setting as per user confirmation
    DEFAULT_OUTPUT_VQA_FILE = "vqa_category4_1.txt"

    # You can modify these to use different file paths if needed.
    input_file = DEFAULT_INPUT_PROMPT_FILE
    output_file = DEFAULT_OUTPUT_VQA_FILE
    
    print(f"Starting VQA generation for Category 4.1 prompts...")
    print(f"Reading prompts from: '{input_file}'")
    print(f"Writing VQA questions to: '{output_file}'")
    
    main_generate_vqa(input_file, output_file)
    
    print(f"✅ VQA questions for Category 4.1 have been written to '{output_file}'") 