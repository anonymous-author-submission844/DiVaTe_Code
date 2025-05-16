import re
from typing import List, Tuple

# === Constants ===
STARTERS = ["A photo of", "A drawing of", "A painting of", "A sketch of", "A picture of"]
LABEL_TYPES = ["a label", "a caption", "a text", "a word"]

def load_sentences(filepath: str) -> List[str]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def extract_category2_3_components(sentence: str) -> Tuple[str, str, str, str, str]:
    starter_regex = '|'.join(re.escape(s) for s in STARTERS)
    label_regex = '|'.join(re.escape(l) for l in LABEL_TYPES)

    # attr1 = single word, obj1 = everything until period
    pattern = rf"({starter_regex}) (a|an)? ?(\w+)\s+(.+?)\. There is ({label_regex}) '([^']+)' written on it\."
    match = re.match(pattern, sentence)

    if match:
        starter = match.group(1)
        attr1 = match.group(3).strip()
        obj1 = match.group(4).strip()
        label_type = match.group(5)
        attr2 = match.group(6).strip()
        return starter, attr1, obj1, label_type, attr2
    else:
        raise ValueError(f"Failed to parse sentence: {sentence}")

def generate_category2_3_vqa(attr1: str, obj1: str, attr2: str) -> List[str]:
    return [
        f"Does this image show a {obj1} primarily made of {attr1}?",
        f"Does the {obj1} appear to be made of {attr2}?",
        f"Focusing on the surroundings (not the {obj1}), do visual textures or other elements suggest the presence of {attr2}?",
        #f"Does the {obj1} in the image appear to be made of {attr2} instead of {attr1}?"
    ]

def generate_category2_3_questions(input_path: str, output_path: str):
    sentences = load_sentences(input_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            try:
                starter, attr1, obj1, label_type, attr2 = extract_category2_3_components(sentence)
                questions = generate_category2_3_vqa(attr1, obj1, attr2)
                f.write(f"Prompt: {sentence}\n")
                for q in questions:
                    f.write(f"{q}\n")
                f.write("\n")
            except ValueError as e:
                print(e)

if __name__ == "__main__":
    input_path = "category2_3.txt"
    output_path = "vqa_category2_3.txt"
    generate_category2_3_questions(input_path, output_path)
    print(f"✅ Category 2-3 VQA questions written to '{output_path}'")
