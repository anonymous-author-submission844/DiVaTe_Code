import random

def generate_category3_6_job(num_prompts=1000, seed=42):
    random.seed(seed)

    starters = ["A photo of", "A drawing of", "A painting of", "A sketch of"]
    label_types = ["a label", "a caption", "a text", "a word"]

    human_objects = {
        "man": ["Doctor", "Police", "Soldier", "Firefighter", "Engineer", "Teacher", "Artist", "Chef", "Pilot", "Librarian", "Gamer", "Astronaut", "Soccer Player", "Basketball Player", "Taxi Driver", "Bus Driver", "Singer"],
        "woman": ["Doctor", "Police", "Soldier", "Firefighter", "Engineer", "Teacher", "Artist", "Chef", "Pilot", "Librarian", "Gamer", "Astronaut", "Soccer Player", "Basketball Player", "Taxi Driver", "Bus Driver", "Singer"],
        "old man": ["Doctor", "Police", "Soldier", "Firefighter", "Engineer", "Teacher", "Artist", "Chef", "Pilot", "Librarian", "Gamer", "Astronaut", "Soccer Player", "Basketball Player", "Taxi Driver", "Bus Driver", "Singer"],
        "old woman": ["Doctor", "Police", "Soldier", "Firefighter", "Engineer", "Teacher", "Artist", "Chef", "Pilot", "Librarian", "Gamer", "Astronaut", "Soccer Player", "Basketball Player", "Taxi Driver", "Bus Driver", "Singer"],
    }

    prompts = []

    while len(prompts) < num_prompts:
        obj = random.choice(list(human_objects.keys()))
        job = random.choice(human_objects[obj])
        starter = random.choice(starters)
        label = random.choice(label_types)

        prompt = f"{starter} {obj}. There is {label} '{job}' written on it."
        if not prompt in prompts:
            prompts.append(prompt)

    return prompts

# Example usage
if __name__ == "__main__":
    prompts = generate_category3_6_job(num_prompts=300)
    with open("category3_3_new.txt", "w") as f:
        for p in prompts:
            f.write(p + "\n")
    print("✅ Job prompts saved to category3_3_new.txt")
