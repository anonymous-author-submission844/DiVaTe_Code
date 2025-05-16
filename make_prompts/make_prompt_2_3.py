import random

texture_to_objects = {
    "Rubber": ["band", "ball", "tire", "gloves", "sole shoes", "eraser", "boots", "mat"],
    "Plastic": ["bottle", "bag", "toy", "cutlery", "chair", "phone case", "container", "cup", "plate"],
    "Metallic": ["car", "jewelry", "watch", "keychain", "desk lamp", "door knob", "spoon", "fork", "knife", "key", "ring", "necklace", "bracelet", "earring"],
    "Wooden": ["chair", "table", "picture frame", "toy", "jewelry box", "door", "floor", "chopsticks", "pencils", "spoon", "knife"],
    "Fabric": ["bag", "pillow", "curtain", "shirt", "pants", "dress", "blanket", "towel", "rug", "hat", "scarf", "sweater", "jacket"],
    "Fluffy": ["pillow", "blanket", "teddy bear", "rug", "sweater", "clouds", "towel", "scarf", "hat"],
    "Leather": ["jacket", "shoes", "belt", "bag", "wallet", "gloves", "chair", "sofa", "hat", "watch"],
    "Glass": ["bottle", "vase", "window", "cup", "mirror", "jar", "table", "bowl", "plate"]
}

# Label types and other parts
label_types = ["a label", "a caption", "a text", "a word"]
starters = ["A picture of", "A photo of", "A drawing of", "A sketch of"]
middle = "There is"
ending = "written on it."

def generate_attribute_conflict_prompts(num_prompts=400, seed=42):
    random.seed(seed)

    # Build all valid (attr1, object) pairs
    attr_object_pairs = [(attr, obj) for attr, objs in texture_to_objects.items() for obj in objs]
    
    prompts = []
    for _ in range(num_prompts):
        attr1, obj = random.choice(attr_object_pairs)

        # Ensure attr2 ≠ attr1
        attr2_choices = [a for a in texture_to_objects if a != attr1]
        attr2 = random.choice(attr2_choices)

        starter = random.choice(starters)
        label = random.choice(label_types)

        prompt = f"{starter} a {attr1.lower()} {obj}. {middle} {label} '{attr2.lower()}' {ending}"
        prompts.append(prompt)

    return prompts

# Example usage
if __name__ == "__main__":
    prompts = generate_attribute_conflict_prompts(num_prompts=300)
    with open("category2_3.txt", "w") as f:
        for p in prompts:
            f.write(p + "\n")
    print("✅ Prompts saved to category2_3.txt")
