import random
from itertools import product

def generate_category2_prompts(num_prompts=400, seed=42):
    random.seed(seed)

    # [1] Appears 25% each
    starters = ["A photo of", "A drawing of", "A painting of", "A sketch of"]

    # [2] & [5] Cartesian sets
    entities = [
        "beaver", "dolphin", "otter", "seal", "whale", "aquarium fish", "flatfish", "ray", "shark", "trout",
        "orchids", "poppies", "roses", "sunflowers", "tulips", "bottles", "bowls", "cans", "cups", "plates",
        "apples", "mushrooms", "oranges", "pears", "sweet peppers", "clock", "computer keyboard", "lamp", 
        "telephone", "television", "bed", "chair", "couch", "table", "wardrobe", "bee", "beetle", "butterfly", 
        "caterpillar", "cockroach", "bear", "leopard", "lion", "tiger", "wolf", "bridge", "castle", "house", 
        "road", "skyscraper", "cloud", "forest", "mountain", "plain", "sea", "camel", "cattle", "chimpanzee", 
        "elephant", "kangaroo", "fox", "porcupine", "possum", "raccoon", "skunk", "crab", "lobster", "snail", 
        "spider", "worm", "baby", "boy", "girl", "man", "woman", "crocodile", "dinosaur", "lizard", "snake", 
        "turtle", "hamster", "mouse", "rabbit", "shrew", "squirrel", "maple", "oak", "palm", "pine", "willow", 
        "bicycle", "bus", "motorcycle", "pickup truck", "train", "lawn-mower", "rocket", "streetcar", "tank", 
        "tractor", "firefighter", "police", "chef", "teacher", "athlete", "student", "soldier", "taxi-driver"
    ]

    # [3] & [6]
    middle = "There is"
    ending = "written on it."

    # [4] 25% chance each
    label_types = ["a label", "a caption", "a text", "a word"]

    # Generate all combinations avoiding same object
    all_pairs = [(a, b) for a in entities for b in entities if a != b]
    random.shuffle(all_pairs)
    selected = all_pairs[:num_prompts]

    # Assemble prompts
    prompts = []
    for entity1, entity2 in selected:
        s = random.choice(starters)
        l = random.choice(label_types)
        prompt = f"{s} {entity1}. {middle} {l} '{entity2}' {ending}"
        prompts.append(prompt)

    return prompts

# Example usage
if __name__ == "__main__":
    prompts = generate_category2_prompts(num_prompts=400)
    with open("category2_1.txt", "w") as f:
        for p in prompts:
            f.write(p + "\n")
    print("✅ Prompts saved to category2_1.txt")
