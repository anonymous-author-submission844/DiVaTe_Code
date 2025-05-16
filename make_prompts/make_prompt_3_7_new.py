import random

def generate_category3_6_brand(num_prompts=1000, seed=42):
    random.seed(seed)

    starters = ["A photo of", "A drawing of", "A painting of", "A sketch of"]
    label_types = ["a label", "a caption", "a text", "a word"]

    brand_objects = {
        "plastic cup": ["Coca-Cola", "Pepsi", "Sprite", "Monster Energy", "Starbucks", "McDonald"],
        "glass cup": ["Coca-Cola", "Pepsi", "Sprite", "Monster Energy", "Blue Bottle Coffee", "Heineken"],
        "mug cup": ["Starbucks", "McDonald", "Blue Bottle Coffee", "Powerade"],
        "can": ["Coca-Cola", "Pepsi", "Monster Energy", "Mountain Dew", "Gatorade"],
        "plastic bottle": ["Coca-Cola", "Pepsi", "Sprite", "Gatorade", "Powerade"]
    }

    prompts = []

    while len(prompts) < num_prompts:
        obj = random.choice(list(brand_objects.keys()))
        brand = random.choice(brand_objects[obj])
        starter = random.choice(starters)
        label = random.choice(label_types)

        prompt = f"{starter} {obj}. There is {label} '{brand}' written on it."
        prompts.append(prompt)

    return prompts

# Example usage
if __name__ == "__main__":
    prompts = generate_category3_6_brand(num_prompts=150)
    with open("category3_7_new.txt", "w") as f:
        for p in prompts:
            f.write(p + "\n")
    print("✅ Brand prompts saved to category3_7_new.txt")
