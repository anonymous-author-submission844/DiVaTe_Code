import random

def generate_structured_prompts(num_prompts=2000, seed=42):
    random.seed(seed)

    # Prompt templates
    starters = ["A photo of", "A drawing of", "A painting of", "A sketch of"]
    label_types = ["a label", "a caption", "a text", "a word"]

    # Attribute categories
    colors = ["black", "red", "blue", "green", "pink", "brown", "yellow", "grey"]
    textures = ["wooden", "metallic"]
    lengths = ["long", "short"]

    # Body parts
    human_parts_color = ["eyes", "hair"]
    human_parts_length = ["hair"]
    human_parts_texture = ["head", "body", "leg", "hand"]

    animal_parts_color = ["eyes", "fur"]
    animal_parts_texture = ["head", "body", "leg", "hand"]

    # Objects
    humans = ["girl", "boy", "man", "woman", "teacher", "soldier", "police", "baby", "chef", "student"]
    animals = ["cat", "dog", "lion", "tiger", "elephant", "rabbit", "bear", "monkey", "wolf", "racoon"]

    prompts = []

    def make_prompt(category_type, starter, label_type):
        if category_type == 1:
            person = random.choice(humans)
            color = random.choice(colors)
            part = random.choice(human_parts_color)
            return f"{starter} a {person}. There is {label_type} '{color} {part}' written on it."

        elif category_type == 2:
            person = random.choice(humans)
            length = random.choice(lengths)
            return f"{starter} a {person}. There is {label_type} '{length} hair' written on it."

        elif category_type == 3:
            person = random.choice(humans)
            texture = random.choice(textures)
            part = random.choice(human_parts_texture)
            return f"{starter} a {person}. There is {label_type} '{texture} {part}' written on it."

        elif category_type == 4:
            animal = random.choice(animals)
            color = random.choice(colors)
            part = random.choice(animal_parts_color)
            return f"{starter} a {animal}. There is {label_type} '{color} {part}' written on it."

        elif category_type == 5:
            animal = random.choice(animals)
            texture = random.choice(textures)
            part = random.choice(animal_parts_texture)
            return f"{starter} a {animal}. There is {label_type} '{texture} {part}' written on it."

    # Generate prompts evenly across 5 categories
    for i in range(num_prompts):
        cat = (i % 5) + 1  # Round-robin from category 1 to 5
        starter = random.choice(starters)
        label_type = random.choice(label_types)
        prompt = make_prompt(cat, starter, label_type)
        if prompt not in prompts:
            prompts.append(prompt)

    return prompts

# Save prompts
if __name__ == "__main__":
    prompts = generate_structured_prompts(num_prompts=400)
    with open("category3_1.txt", "w") as f:
        for p in prompts:
            f.write(p + "\n")
    print("✅ Structured prompts saved to category3_1.txt")
