import re
from typing import List, Tuple

# === Constants ===
STARTERS = ["A photo of", "A drawing of", "A painting of", "A sketch of", "A picture of"]
LABEL_TYPES = ["a label", "a caption", "a text", "a word"]

def load_sentences(filepath: str) -> List[str]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def extract_category3_1_components(sentence: str) -> Tuple[str, str, str, str, str]:
    starter_regex = '|'.join(re.escape(s) for s in STARTERS)
    label_regex = '|'.join(re.escape(l) for l in LABEL_TYPES)

    # obj1 = object before period, label = 'attr1 part'
    pattern = rf"({starter_regex}) (a|an)? ?(.+?)\. There is ({label_regex}) '([^']+)' written on it\."
    match = re.match(pattern, sentence)

    if match:
        starter = match.group(1)
        obj1 = match.group(3).strip()
        label_type = match.group(4)
        label_content = match.group(5).strip()

        # Split label into attr1 and part
        parts = label_content.split(" ", 1)
        if len(parts) != 2:
            raise ValueError(f"Label must have two parts: attr + part: {label_content}")
        attr1, part = parts[0], parts[1]

        return starter, obj1, label_type, attr1, part
    else:
        raise ValueError(f"Failed to parse sentence: {sentence}")

def generate_category3_1_vqa(obj1: str, attr1: str, part: str) -> List[str]:
    return [
        f"Does this image primarily depict a {obj1}?",
        f"Does the {obj1} in the image visually exhibit the feature {attr1} {part}?",
        f"Apart from the {obj1}'s direct features, does the image's overall style or focus seem related to the concept of {attr1}?"
    ]

def generate_category3_1_questions(input_path: str, output_path: str):
    sentences = load_sentences(input_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            try:
                starter, obj1, label_type, attr1, part = extract_category3_1_components(sentence)
                questions = generate_category3_1_vqa(obj1, attr1, part)
                f.write(f"Prompt: {sentence}\n")
                for q in questions:
                    f.write(f"{q}\n")
                f.write("\n")
            except ValueError as e:
                print(e)

if __name__ == "__main__":
    input_path = "category3_1.txt"
    output_path = "vqa_category3_1.txt"
    generate_category3_1_questions(input_path, output_path)
    print(f"✅ Category 3-1 VQA questions written to '{output_path}'")
