import random

def generate_category4_2_spatial_prompts(num_prompts=300, seed=42):
    random.seed(seed)

    starters = ["A photo of", "A drawing of", "A painting of", "A sketch of"]
    label_types = ["a label", "a caption", "a text", "a word"]

    # Movable and static object separation
    movable_objects = [
        "beaver", "dolphin", "otter", "seal", "whale", "trout", "bee", "butterfly", "bear",
        "leopard", "lion", "tiger", "wolf", "camel", "elephant", "kangaroo", "fox", "raccoon",
        "crab", "lobster", "snail", "spider", "worm", "baby", "boy", "girl", "man", "woman",
        "crocodile", "dinosaur", "lizard", "snake", "turtle", "rabbit", "squirrel", "bicycle",
        "bus", "motorcycle", "truck", "train", "rocket", "tank", "firefighter", "police", "chef",
        "teacher", "athlete", "student", "soldier", "taxi-driver"
    ]

    static_objects = [
        "rose", "sunflower", "tulip", "apple", "mushroom", "pear", "bottle", "bowl", "can", "cup",
        "plate", "clock", "lamp", "chair", "couch", "table", "wardrobe", "castle", "house", "cloud",
        "mountain", "maple", "oak", "pine"
    ]

    all_objects = movable_objects + static_objects

    # Heading-type direction pairs (require movement)
    heading_directions = [("east", "west"), ("west", "east")]

    # Positional directions (okay for static or any object)
    positional_directions = [
        ("front", "back"), ("back", "front"),
        ("left", "right"), ("right", "left"),
        ("up", "down"), ("down", "up"),
        ("top", "bottom"), ("bottom", "top")
    ]

    prompts = []

    while len(prompts) < num_prompts:
        starter = random.choice(starters)
        label_type = random.choice(label_types)
        obj = random.choice(all_objects)

        if obj in movable_objects:
            direction_pairs = heading_directions + positional_directions
        else:
            direction_pairs = positional_directions  # exclude heading

        dir1, dir2 = random.choice(direction_pairs)

        # Format prompt depending on direction type
        if (dir1, dir2) in heading_directions:
            prompt = (
                f"{starter} two {obj}s, one {obj} is heading {dir1} and the other is heading {dir2}. "
                f"The one heading {dir1} has {label_type} 'heading {dir2}', and the one heading {dir2} has "
                f"{label_type} 'heading {dir1}' on it."
            )
        else:
            prompt = (
                f"{starter} two {obj}s, one {obj} is on the {dir1} and the other is on the {dir2}. "
                f"The one on the {dir1} has {label_type} '{dir2}', and the one on the {dir2} has "
                f"{label_type} '{dir1}' on it."
            )

        prompts.append(prompt)

    return prompts


# Save prompts to file
if __name__ == "__main__":
    prompts = generate_category4_2_spatial_prompts(num_prompts=300)
    with open("category4_2_spatial.txt", "w") as f:
        for p in prompts:
            f.write(p + "\n")
    print("✅ Prompts saved to category4_2_spatial.txt")
