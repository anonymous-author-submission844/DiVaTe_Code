import re
from typing import List, Tuple

# === Constants ===
STARTERS = ["A photo of", "A drawing of", "A painting of", "A sketch of", "A picture of"]
LABEL_TYPES = ["a label", "a caption", "a text", "a word"]

def load_sentences(filepath: str) -> List[str]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def extract_category3_6_components(sentence: str) -> Tuple[str, str, str, str, str]:
    starter_regex = '|'.join(re.escape(s) for s in STARTERS)
    label_regex = '|'.join(re.escape(l) for l in LABEL_TYPES)

    # Support both "written on it." and "written between them."
    pattern = rf"({starter_regex}) (a|an)? ?(.+?) and (.+?)\. There is ({label_regex}) '([^']+)' written (on it|between them)\."
    match = re.match(pattern, sentence)

    if match:
        starter = match.group(1)
        obj1 = match.group(3).strip()
        obj2 = match.group(4).strip()
        label_type = match.group(5)
        relationship = match.group(6).strip()
        return starter, obj1, obj2, label_type, relationship
    else:
        raise ValueError(f"Failed to parse sentence: {sentence}")

def generate_category3_6_vqa(obj1: str, obj2: str, relationship: str) -> List[str]:
    return [
        f"Does this image primarily depict both {obj1} and {obj2}?",
        f"Do the poses or interactions of {obj1} and {obj2} visually depict them as {relationship}?",
        f"Does the scene's overall mood or composition subtly suggest the relationship {relationship}?"
    ]

def generate_category3_6_questions(input_path: str, output_path: str):
    sentences = load_sentences(input_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            try:
                starter, obj1, obj2, label_type, relationship = extract_category3_6_components(sentence)
                questions = generate_category3_6_vqa(obj1, obj2, relationship)
                f.write(f"Prompt: {sentence}\n")
                for q in questions:
                    f.write(f"{q}\n")
                f.write("\n")
            except ValueError as e:
                print(e)


if __name__ == "__main__":
    input_path = "category3_6.txt"
    output_path = "vqa_category3_6.txt"
    generate_category3_6_questions(input_path, output_path)
    print(f"✅ Category 3-6 VQA questions written to '{output_path}'")
