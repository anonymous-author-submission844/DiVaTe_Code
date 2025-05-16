import random
from itertools import combinations

def generate_category2_2_conflict_prompts(num_prompts_per_type=700, seed=42):
    random.seed(seed)

    starters = ["A photo of", "A drawing of", "A painting of", "A sketch of"]
    middle = "There is"
    label_types = ["a label", "a caption", "a text", "a word"]
    ending = "written on it."

    # Entities that can sensibly have shape
    shape_entities = [
        "watch", "table", "mirror", "water bottle", "bowl", "plate", "cup", "bed"
    ]

    # Entities that can sensibly have color
    color_entities = [
        "chair", "table", "couch", "cup", "bottle", "car", "bus", "train", "lamp",
        "plate", "bowl", "shirt", "pants", "dress", "shoes", "vase", "phone case", "bed", "mirror", "toy"
    ]

    # Attribute sets
    shapes = ["circle", "triangle", "rectangle", "star", "ellipse", "square"]
    colors = ["black", "pink", "blue", "yellow", "green", "red", "white"]

    def get_conflict_pairs(attributes):
        return list(combinations(attributes, 2))

    def generate_from_conflicts(attr_list, valid_entities):
        pairs = get_conflict_pairs(attr_list)
        random.shuffle(pairs)
        prompts = []

        while len(prompts) < num_prompts_per_type:
            attr1, attr2 = random.choice(pairs)
            entity = random.choice(valid_entities)
            starter = random.choice(starters)
            label_type = random.choice(label_types)

            prompt = f"{starter} a {attr1} {entity}. {middle} {label_type} '{attr2}' {ending}"
            prompts.append(prompt)

        return prompts

    shape_prompts = generate_from_conflicts(shapes, shape_entities)
    color_prompts = generate_from_conflicts(colors, color_entities)

    all_prompts = shape_prompts + color_prompts
    random.shuffle(all_prompts)
    return all_prompts

# Example usage
if __name__ == "__main__":
    prompts = generate_category2_2_conflict_prompts(num_prompts_per_type=300)
    with open("category2_2.txt", "w") as f:
        for p in prompts:
            f.write(p + "\n")
    print("✅ Prompts saved to category2_2.txt")
