import random
from itertools import product

def generate_category1_prompts(num_prompts=2000, seed=42):
    random.seed(seed)

    # [1] Appears with 25% probability each
    starters = ["A photo of", "A drawing of", "A painting of", "A sketch of"]

    # [2] Cartesian product with target labels
    surfaces = [
        "a blank wall", "an empty canvas", "a chalkboard", "a whiteboard", "an A4 paper",
        "a poster board", "a notebook page", "a signboard", "a blackboard", "a billboard"
    ]

    # [3] Always appears
    middle = "There is"

    # [4] Appears with 25% probability each
    label_types = ["a label", "a caption", "a text", "a word"]

    # [5] Cartesian product targets (things to be labeled)
    targets = [
        "beaver", "dolphin", "otter", "seal", "whale", "aquarium fish", "flatfish", "ray", "shark", "trout",
        "orchids", "poppies", "roses", "sunflowers", "tulips",
        "bottles", "bowls", "cans", "cups", "plates", "apples", "mushrooms", "oranges", "pears", "sweet peppers",
        "clock", "computer keyboard", "lamp", "telephone", "television", "bed", "chair", "couch", "table", "wardrobe",
        "bee", "beetle", "butterfly", "caterpillar", "cockroach", "bear", "leopard", "lion", "tiger", "wolf",
        "bridge", "castle", "house", "road", "skyscraper", "cloud", "forest", "mountain", "plain", "sea",
        "camel", "cattle", "chimpanzee", "elephant", "kangaroo", "fox", "porcupine", "possum", "raccoon", "skunk",
        "crab", "lobster", "snail", "spider", "worm", "baby", "boy", "girl", "man", "woman",
        "crocodile", "dinosaur", "lizard", "snake", "turtle", "hamster", "mouse", "rabbit", "shrew", "squirrel",
        "maple", "oak", "palm", "pine", "willow",
        "bicycle", "bus", "motorcycle", "pickup truck", "train", "lawn-mower", "rocket", "streetcar", "tank", "tractor",
        "firefighter", "police", "chef", "teacher", "athlete", "student", "soldier", "taxi-driver"
    ]

    # [6] Fixed ending
    ending = "written on it."

    # Cartesian product over [2] and [5]
    all_combinations = list(product(surfaces, targets))
    random.shuffle(all_combinations)

    # Limit the number of prompts
    selected = all_combinations[:num_prompts]

    prompts = []
    for surface, target in selected:
        s = random.choice(starters)
        l = random.choice(label_types)
        prompt = f"{s} {surface}. {middle} {l} '{target}' {ending}"
        prompts.append(prompt)

    return prompts

# Example usage:
if __name__ == "__main__":
    prompts = generate_category1_prompts(num_prompts=1000)
    with open("category1.txt", "w") as f:
        for p in prompts:
            f.write(p + "\n")
    print("Prompts saved to category1.txt")
