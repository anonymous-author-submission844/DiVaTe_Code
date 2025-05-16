import re
from typing import List, Tuple

# === Constants ===
STARTERS = ["A photo of", "A drawing of", "A painting of", "A sketch of", "A picture of"]
LABEL_TYPES = ["a label", "a caption", "a text", "a word"]
PREPOSITIONS = ["wearing", "on the", "below the", "next to"]

def load_sentences(filepath: str) -> List[str]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def extract_category3_4_components(sentence: str) -> Tuple[str, str, str, str]:
    starter_regex = '|'.join(re.escape(s) for s in STARTERS)
    label_regex = '|'.join(re.escape(l) for l in LABEL_TYPES)

    pattern = rf"({starter_regex}) (a|an)? ?(.+?)\. There is ({label_regex}) '([^']+)' written on it\."
    match = re.match(pattern, sentence)

    if match:
        starter = match.group(1)
        obj1 = match.group(3).strip()
        label_type = match.group(4)
        label_text = match.group(5).strip()

        # Match preposition + obj2 from label_text
        for prep in PREPOSITIONS:
            if label_text.lower().startswith(prep):
                obj2 = label_text[len(prep):].strip()
                return obj1, prep, obj2

        raise ValueError(f"Could not identify preposition in label: '{label_text}'")

    else:
        raise ValueError(f"Failed to parse sentence: {sentence}")

def generate_category3_4_vqa(obj1: str, prep: str, obj2: str) -> List[str]:
    return [
        f"Does this image primarily depict a {obj1}?",
        f"Is the {obj1} {prep} {obj2}?",
        f"Beyond the {obj1}'s direct placement, does the broader environment of the scene seem to visually incorporate elements of {prep} {obj2}?"
    ]

def generate_category3_4_questions(input_path: str, output_path: str):
    sentences = load_sentences(input_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            try:
                obj1, prep, obj2 = extract_category3_4_components(sentence)
                questions = generate_category3_4_vqa(obj1, prep, obj2)
                f.write(f"Prompt: {sentence}\n")
                for q in questions:
                    f.write(f"{q}\n")
                f.write("\n")
            except ValueError as e:
                print(e)

if __name__ == "__main__":
    input_path = "category3_4.txt"
    output_path = "vqa_category3_4.txt"
    generate_category3_4_questions(input_path, output_path)
    print(f"✅ Category 3-4 VQA questions written to '{output_path}'")
