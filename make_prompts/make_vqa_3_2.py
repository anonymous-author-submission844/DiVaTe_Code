import re
from typing import List, Tuple

# === Constants ===
STARTERS = ["A photo of", "A drawing of", "A painting of", "A sketch of", "A picture of"]
LABEL_TYPES = ["a label", "a caption", "a text", "a word"]

def load_sentences(filepath: str) -> List[str]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def extract_category3_2_components(sentence: str) -> Tuple[str, str, str, str]:
    starter_regex = '|'.join(re.escape(s) for s in STARTERS)
    label_regex = '|'.join(re.escape(l) for l in LABEL_TYPES)

    # Format: A sketch of X. There is a label 'Y' written on it.
    pattern = rf"({starter_regex}) (a|an)? ?(.+?)\. There is ({label_regex}) '([^']+)' written on it\."
    match = re.match(pattern, sentence)

    if match:
        starter = match.group(1)
        obj1 = match.group(3).strip()
        label_type = match.group(4)
        action_or_emotion = match.group(5).strip()
        return starter, obj1, label_type, action_or_emotion
    else:
        raise ValueError(f"Failed to parse sentence: {sentence}")

def generate_category3_2_vqa(obj1: str, action_or_emotion: str) -> List[str]:
    return [
        f"Does this image primarily depict a {obj1}?",
        f"Is the {obj1} in the image visually depicted in the state of {action_or_emotion}?",
        f"Does the scene's overall mood or composition seem influenced by the concept of {action_or_emotion}, beyond the {obj1}?"
    ]

def generate_category3_2_questions(input_path: str, output_path: str):
    sentences = load_sentences(input_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            try:
                starter, obj1, label_type, action_or_emotion = extract_category3_2_components(sentence)
                questions = generate_category3_2_vqa(obj1, action_or_emotion)
                f.write(f"Prompt: {sentence}\n")
                for q in questions:
                    f.write(f"{q}\n")
                f.write("\n")
            except ValueError as e:
                print(e)

if __name__ == "__main__":
    input_path = "category3_2.txt"
    output_path = "vqa_category3_2.txt"
    generate_category3_2_questions(input_path, output_path)
    print(f"✅ Category 3-2 VQA questions written to '{output_path}'")
