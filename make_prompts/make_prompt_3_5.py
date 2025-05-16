import random

def generate_category4_5_prompts(num_prompts=2000, seed=42):
    random.seed(seed)

    # [1] Starter phrases
    starters = ["A photo of", "A drawing of", "A painting of", "A sketch of"]

    # [2] Label type phrases
    label_types = ["a label", "a caption", "a text", "a word"]

    # [object] All relevant objects (same as 4-4)
    objects = [
        "baby", "boy", "girl", "man", "woman", "firefighter", "police", "chef", 
        "teacher", "athlete", "student", "soldier", "taxi-driver",
        "beaver", "dolphin", "otter", "seal", "whale", "aquarium fish", "flatfish", 
        "ray", "shark", "trout", "bee", "beetle", "butterfly", "caterpillar", 
        "cockroach", "bear", "leopard", "lion", "tiger", "wolf", "camel", "cattle", 
        "chimpanzee", "elephant", "kangaroo", "fox", "porcupine", "possum", 
        "raccoon", "skunk", "crab", "lobster", "snail", "spider", "worm", 
        "crocodile", "dinosaur", "lizard", "snake", "turtle", "hamster", "mouse", 
        "rabbit", "shrew", "squirrel"
    ]

    # [Attr1] Environment or location labels
    environments = [
        "cloud", "forest", "mountain", "plain", "sea",
        "bridge", "castle", "house", "road", "skyscraper",
        "river", "office"
    ]

    prompts = []
    while len(prompts) < num_prompts:
        starter = random.choice(starters)
        label_type = random.choice(label_types)
        obj = random.choice(objects)
        context = random.choice(environments)

        prompt = f"{starter} {obj}. There is {label_type} '{context}' written on it."
        prompts.append(prompt)

    return prompts

# Example usage
if __name__ == "__main__":
    prompts = generate_category4_5_prompts(num_prompts=300)
    with open("category3_5.txt", "w") as f:
        for p in prompts:
            f.write(p + "\n")
    print("✅ Prompts saved to category3_5.txt")
