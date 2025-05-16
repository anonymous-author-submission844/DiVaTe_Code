import re
from typing import List, Tuple

# === Constants ===
STARTERS = ["A photo of", "A drawing of", "A painting of", "A sketch of"]
LABEL_TYPES = ["a label", "a caption", "a text", "a word"]

# === Step 1: Read lines from file ===
def load_sentences(filepath: str) -> List[str]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

# === Step 2: Extract components from sentence ===
def extract_components(sentence: str) -> Tuple[str, str, str, str]:
    starter_regex = '|'.join(re.escape(s) for s in STARTERS)
    label_regex = '|'.join(re.escape(l) for l in LABEL_TYPES)

    # Updated pattern to support "a" or "an" before neutral_obj
    pattern = rf"({starter_regex}) (a|an) ([^\.]+)\. There is ({label_regex}) '([^']+)' written on it\."
    match = re.match(pattern, sentence)

    if match:
        starter = match.group(1)
        neutral_obj = match.group(3)
        label_type = match.group(4)
        obj = match.group(5)
        return starter, neutral_obj, label_type, obj
    else:
        raise ValueError(f"Failed to parse sentence: {sentence}")

# === Step 3: Generate VQA questions ===
def generate_vqa_questions(neutral_obj: str, obj: str) -> List[str]:
    return [
        f"Does this image primarily depict {neutral_obj}?",
        f"Does the image show any visual features (e.g., texture, pattern) of a {obj}?",
        f"Does the image's overall style or background seem related to a {obj}?"
    ]

# === Step 4: Full pipeline with file output ===
def generate_questions_to_file(input_path: str, output_path: str):
    sentences = load_sentences(input_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            try:
                starter, neutral_obj, label_type, obj = extract_components(sentence)
                questions = generate_vqa_questions(neutral_obj, obj)
                f.write(f"Prompt: {sentence}\n")
                for q in questions:
                    f.write(f"{q}\n")
                f.write("\n")
            except ValueError as e:
                print(e)

# === Run script ===
if __name__ == "__main__":
    input_path = "category1.txt"          # <- Replace with your file if needed
    output_path = "vqa_category1.txt"
    generate_questions_to_file(input_path, output_path)
    print(f"✅ VQA questions written to '{output_path}'")
