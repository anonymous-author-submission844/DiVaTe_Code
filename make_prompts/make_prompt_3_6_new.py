import random

def generate_category4_3_prompts(num_prompts=2000, seed=42):
    random.seed(seed)

    # [1] 25% chance
    starters = ["A photo of", "A drawing of", "A painting of", "A sketch of"]

    # [2] 25% chance
    label_types = ["a label", "a caption", "a text", "a word"]

    # [3] positioning
    position_types = ["written on it", "written between them"]

    # Human objects
    humans = [
        "baby", "boy", "girl", "man", "woman", "firefighter", "police", "chef",
        "teacher", "athlete", "student", "soldier", "taxi-driver"
    ]
    # Animal objects
    animals = [
        "beaver", "dolphin", "otter", "seal", "whale", "aquarium fish", "flatfish", "ray", "shark", "trout",
        "bee", "beetle", "butterfly", "caterpillar", "cockroach", "bear", "leopard", "lion", "tiger", "wolf",
        "camel", "cattle", "chimpanzee", "elephant", "kangaroo", "fox", "porcupine", "possum", "raccoon",
        "skunk", "crab", "lobster", "snail", "spider", "worm", "crocodile", "dinosaur", "lizard", "snake",
        "turtle", "hamster", "mouse", "rabbit", "shrew", "squirrel"
    ]

    # Relationships
    human_relationships_all = ["fighting", "arguing"]
    human_relationships_adults = ["married", "parent and child"]
    animal_relationships_all = ["fighting", "staring"]
    animal_relationships_same = ["parent and child"]

    prompts = []

    def is_child(obj):
        return obj in {"baby", "boy", "girl"}

    while len(prompts) < num_prompts:
        starter = random.choice(starters)
        label = random.choice(label_types)
        position = random.choice(position_types)

        if random.random() < 0.5:  # Human pair
            obj1, obj2 = random.sample(humans, 2)

            valid_rels = human_relationships_all[:]
            if not (is_child(obj1) or is_child(obj2)):
                valid_rels += human_relationships_adults

            relationship = random.choice(valid_rels)

        else:  # Animal pair
            obj1, obj2 = random.sample(animals, 2)

            valid_rels = animal_relationships_all[:]
            if obj1 == obj2:
                valid_rels += animal_relationships_same

            relationship = random.choice(valid_rels)

        prompt = f"{starter} {obj1} and {obj2}. There is {label} '{relationship}' {position}."
        prompts.append(prompt)

    return prompts

# Example usage
if __name__ == "__main__":
    prompts = generate_category4_3_prompts(num_prompts=150)
    with open("category3_6_new.txt", "w") as f:
        for p in prompts:
            f.write(p + "\n")
    print("✅ Prompts saved to category3_6_new.txt")
