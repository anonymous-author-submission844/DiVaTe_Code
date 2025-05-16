import random

def generate_category3_4_prompts(num_prompts=2000, seed=42):
    random.seed(seed)

    starters = ["A photo of", "A drawing of", "A painting of", "A sketch of"]
    label_types = ["a label", "a caption", "a text", "a word"]

    # "Wearing" context
    accessories = [
        "hat", "sneakers", "boots", "jackets", "watch", "necklace", "glasses",
        "sunglasses", "baseball cap", "hoodie"
    ]

    # "On" and "below" surfaces
    on_surfaces = [
        "the rock", "the table", "the chair", "the couch", "the floor", "the shelf",
        "the bench", "the bed", "the ground", "the grass", "the sand", "the snow"
    ]

    # "Next to" context
    attached_objects = [
        "the tree", "the car", "the house", "the wall", "the fence", "the trash can",
        "the building", "the door", "the window", "the truck", "the bench"
    ]

    # Primary objects
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

    prompts = []
    for _ in range(num_prompts):
        starter = random.choice(starters)
        label_type = random.choice(label_types)
        obj = random.choice(objects)

        p = random.random()
        if p < 0.4:
            # 40% chance: wearing
            item = random.choice(accessories)
            attr = f"wearing {item}"
        elif p < 0.6:
            # 20% chance: on
            surface = random.choice(on_surfaces)
            attr = f"on {surface}"
        elif p < 0.8:
            # 20% chance: below
            surface = random.choice(on_surfaces)
            attr = f"below {surface}"
        else:
            # 20% chance: next to
            obj_attached = random.choice(attached_objects)
            attr = f"next to {obj_attached}"

        prompt = f"{starter} {obj}. There is {label_type} '{attr}' written on it."
        prompts.append(prompt)

    return prompts

# Example usage
if __name__ == "__main__":
    prompts = generate_category3_4_prompts(num_prompts=300)
    with open("category3_4.txt", "w") as f:
        for p in prompts:
            f.write(p + "\n")
    print("✅ Prompts saved to category3_4.txt")
