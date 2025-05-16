import re
from typing import List, Tuple

# === Constants ===
STARTERS = ["A photo of", "A drawing of", "A painting of", "A sketch of"]
LABEL_TYPES = ["a label", "a caption", "a text", "a word"]

# === Step 1: Read lines ===
def load_sentences(filepath: str) -> List[str]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

# === Step 2: Extract obj1 and obj2 ===
def extract_category2_components(sentence: str) -> Tuple[str, str, str, str]:
    starter_regex = '|'.join(re.escape(s) for s in STARTERS)
    label_regex = '|'.join(re.escape(l) for l in LABEL_TYPES)

    # Make "a"/"an" optional before the object
    pattern = rf"({starter_regex}) (a|an)? ?([^\.]+)\. There is ({label_regex}) '([^']+)' written on it\."
    match = re.match(pattern, sentence)

    if match:
        starter = match.group(1)
        obj1 = match.group(3).strip()
        label_type = match.group(4)
        obj2 = match.group(5)
        return starter, obj1, label_type, obj2
    else:
        raise ValueError(f"Failed to parse sentence: {sentence}")

# === Step 3: Generate VQA questions ===
def generate_category2_vqa(obj1: str, obj2: str) -> List[str]:
    return [
        f"Does this image primarily depict a {obj1}?",
        f"Is a {obj2} visually present in the image, potentially altering or replacing the intended {obj1}?",
        f"Apart from the main subject, does the background or overall scene contain visual cues or features related to a {obj2}?",
        #f"Does the image predominantly show a {obj2} as the main subject?"
    ]

# === Step 4: Full pipeline with file output ===
def generate_category2_questions(input_path: str, output_path: str):
    sentences = load_sentences(input_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            try:
                starter, obj1, label_type, obj2 = extract_category2_components(sentence)
                questions = generate_category2_vqa(obj1, obj2)
                f.write(f"Prompt: {sentence}\n")
                for q in questions:
                    f.write(f"{q}\n")
                f.write("\n")
            except ValueError as e:
                print(e)

# === Run ===
if __name__ == "__main__":
    input_path = "category2_1.txt"           # Replace with actual file path
    output_path = "vqa_category2_1.txt"
    generate_category2_questions(input_path, output_path)
    print(f"✅ Category 2-1 VQA questions written to '{output_path}'")
