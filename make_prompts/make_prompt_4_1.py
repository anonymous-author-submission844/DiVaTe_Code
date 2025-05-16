import random

def generate_category4_1_quantity_mixed(num_prompts=300, seed=42):
    random.seed(seed)

    starters = ["A photo of", "A drawing of", "A painting of", "A sketch of"]
    label_types = ["a label", "a caption", "a text", "a word"]

    object_label_dict = {
        "bottle": [("10L", "20L"), ("small", "big"), ("300ml", "500ml"), ("1L", "3L"), ("100ml", "4L")],
        "glass": [("10L", "20L"), ("small", "big"), ("300ml", "500ml"), ("1L", "3L"), ("100ml", "4L")],
        "cup": [("10L", "20L"), ("small", "big"), ("300ml", "500ml"), ("1L", "3L"), ("100ml", "4L")],
        "box": [("1kg", "3kg"), ("2kg", "5kg"), ("light", "heavy"), ("100g", "1kg"), ("1kg", "1ton")],
        "stone": [("1kg", "3kg"), ("2kg", "5kg"), ("light", "heavy"), ("100g", "1kg"), ("1kg", "1ton")],
        "wooden ball": [("1kg", "3kg"), ("2kg", "5kg"), ("light", "heavy"), ("100g", "1kg"), ("1kg", "1ton")],
        "candle": [("15cm", "30cm"), ("14inch", "27inch"), ("10cm", "1m"), ("short", "long")],
        "ruler": [("15cm", "30cm"), ("14inch", "27inch"), ("10cm", "1m"), ("short", "long")]
    }

    attribute_type_map = {
        "heavier": ["kg", "g", "ton", "light", "heavy"],
        "longer": ["cm", "inch", "m", "short", "long"],
        "larger": ["L", "ml", "big", "small"]
    }

    comparative_to_noun = {
        "heavier": "weight",
        "longer": "length",
        "larger": "size"
    }

    def get_attribute_type(label):
        for key, values in attribute_type_map.items():
            if any(k in label.lower() for k in values):
                return key
        return "larger"

    def pluralize(word):
        if word in ["glass"]:
            return "glasses"
        elif word.endswith("y") and not word.endswith("ey"):
            return word[:-1] + "ies"
        elif word.endswith(("s", "x", "ch", "sh")):
            return word + "es"
        else:
            return word + "s"

    prompts = []
    while len(prompts) < num_prompts:
        obj = random.choice(list(object_label_dict.keys()))
        label1, label2 = random.choice(object_label_dict[obj])
        starter = random.choice(starters)
        label_type = random.choice(label_types)
        attr_type = get_attribute_type(label1)
        noun_attr = comparative_to_noun[attr_type]
        plural_obj = pluralize(obj)

        if len(prompts) < num_prompts // 2:
            # Identical case
            less_term = "lighter" if attr_type == "heavier" else "shorter" if attr_type == "longer" else "smaller"
            prompt = (
                f"{starter} two identical {plural_obj} next to each other, which have same {noun_attr}. "
                f"The left one has {label_type} '{label1}', and the right one has {label_type} '{label2}' on it."
            )
        else:
            # Non-identical case
            less_term = "lighter" if attr_type == "heavier" else "shorter" if attr_type == "longer" else "smaller"
            prompt = (
                f"{starter} two {plural_obj} next to each other, where the left one is {attr_type} than the right one. "
                f"The {attr_type} one has {label_type} '{label1}', and the {less_term} one has {label_type} '{label2}' on it."
            )

        prompts.append(prompt)

    return prompts

# Example usage
if __name__ == "__main__":
    prompts = generate_category4_1_quantity_mixed(num_prompts=300)
    with open("category4_1_quantity.txt", "w") as f:
        for p in prompts:
            f.write(p + "\n")
    print("✅ Prompts saved to category4_1_quantity.txt")
