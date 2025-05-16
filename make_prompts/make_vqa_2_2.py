import re
from typing import List, Tuple

# === Constants ===
STARTERS = ["A photo of", "A drawing of", "A painting of", "A sketch of", "A picture of"]
LABEL_TYPES = ["a label", "a caption", "a text", "a word"]

def load_sentences(filepath: str) -> List[str]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def extract_category2_2_components(sentence: str) -> Tuple[str, str, str, str, str]:
    starter_regex = '|'.join(re.escape(s) for s in STARTERS)
    label_regex = '|'.join(re.escape(l) for l in LABEL_TYPES)

    # attr1: first word, obj1: everything else until "."
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

def generate_category2_2_vqa(attr1: str, obj1: str, attr2: str) -> List[str]:
    return [
        f"Does this image show a {obj1} that is primarily {attr1}?",
        f"Does the {obj1} visually exhibit '{attr2}' (e.g., as its color or shape), instead of or mixed with the expected {attr1} characteristic?",
        #f"Is the {obj1} depicted as being {attr2}?", # or does it show a conflicting mix of {attr1} and {attr2} characteristics?",
        #f"Does the {obj1} show a mix of both {attr1} and {attr2} characteristics, rather than being clearly {attr1}?",
        f"Apart from the {obj1}, is the scene's color palette or style significantly influenced by {attr2}?",
        #f"Is the {obj1} in the image depicted as being {attr2}, instead of {attr1}?",
    ]

def generate_category2_2_questions(input_path: str, output_path: str):
    sentences = load_sentences(input_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            try:
                starter, attr1, obj1, label_type, attr2 = extract_category2_2_components(sentence)
                questions = generate_category2_2_vqa(attr1, obj1, attr2)
                f.write(f"Prompt: {sentence}\n")
                for q in questions:
                    f.write(f"{q}\n")
                f.write("\n")
            except ValueError as e:
                print(e)
# === Run ===
if __name__ == "__main__":
    input_path = "category2_2.txt"         # <- your file here
    output_path = "vqa_category2_2.txt"
    generate_category2_2_questions(input_path, output_path)
    print(f"✅ Category 2-2 VQA questions written to '{output_path}'")
