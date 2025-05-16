import random

def pluralize(word, count):
    if count == 1:
        return word
    elif word.endswith("y") and not word.endswith("ey"):
        return word[:-1] + "ies"
    elif word.endswith(("s", "x", "ch", "sh")):
        return word + "es"
    else:
        return word + "s"

def generate_category4_2_prompts(num_prompts=2000, seed=42):
    random.seed(seed)

    starters = ["A photo of", "A drawing of", "A painting of", "A sketch of"]
    label_types = ["a label", "a caption", "a text", "a word"]

    objects = [
        "apple", "banana", "cookie", "donut", "cupcake", "carrot", "bottle",
        "book", "pen", "chair", "shoe", "teddy bear", "ball", "jelly", "soccer ball",
        "cup", "mug", "bowl", "plate", "clock", "candle", "plant", "toy", "box"
    ]

    prompts = []
    while len(prompts) < num_prompts:
        starter = random.choice(starters)
        label_type = random.choice(label_types)
        obj = random.choice(objects)

        n1 = random.randint(1, 10)
        n2 = random.randint(1, 10)
        while n2 == n1:
            n2 = random.randint(1, 10)

        object1 = pluralize(obj, n1)
        object2 = pluralize(obj, n2)

        prompt = f"{starter} {n1} {object1}. There is {label_type} '{n2} {object2}' written on it."
        prompts.append(prompt)

    return prompts

# Example usage
if __name__ == "__main__":
    prompts = generate_category4_2_prompts(num_prompts=300)
    with open("category4_3.txt", "w") as f:
        for p in prompts:
            f.write(p + "\n")
    print("✅ Prompts saved to category4_3.txt")
