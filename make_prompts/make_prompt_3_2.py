import random

def generate_category4_2_prompts(num_prompts=2000, seed=42):
    random.seed(seed)

    # [1] Appears 25% each
    starters = ["A photo of", "A drawing of", "A painting of", "A sketch of"]

    # [4] Appears 25% each
    label_types = ["a label", "a caption", "a text", "a word"]

    # Action/Expression attributes
    attributes = [
        "running", "sleeping", "standing", "lying", "jumping", "working",
        "angry", "happy", "excited", "nervous", "bored", "scared", "sad"
    ]

    # Object pool
    objects = [
        "baby", "boy", "girl", "man", "woman", "firefighter", "police", "chef",
        "teacher", "athlete", "student", "soldier", "taxi-driver"
    ]

    # Generate prompts
    prompts = []
    while len(prompts) < num_prompts:
        starter = random.choice(starters)
        obj = random.choice(objects)
        label_type = random.choice(label_types)
        attr = random.choice(attributes)

        prompt = f"{starter} {obj}. There is {label_type} '{attr}' written on it."
        prompts.append(prompt)

    return prompts

# Example usage
if __name__ == "__main__":
    prompts = generate_category4_2_prompts(num_prompts=300)
    with open("category3_2.txt", "w") as f:
        for p in prompts:
            f.write(p + "\n")
    print("✅ Prompts saved to category3_2.txt")
