import json
import random
import os

random.seed(42)

def main():
    input_path = "data/squad_qa.json"
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    with open(input_path, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} items from {input_path}")
    
    # Shuffle for random split
    random.shuffle(data)

    train_data = data[:80000]
    test_data = data[80000:]

    with open("data/squad_train_80k.json", "w") as f:
        json.dump(train_data, f, indent=2)

    with open("data/squad_test_6k.json", "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"Saved {len(train_data)} to squad_train_80k.json")
    print(f"Saved {len(test_data)} to squad_test_6k.json")

    # Also output a pure text version of the test sentences for the test_brain.cpp Predictor Held-Out test
    with open("data/test_sentences_6k.txt", "w") as f:
        for item in test_data:
            f.write(item["input"].replace("\n", " ") + " " + item["target"].replace("\n", " ") + "\n")
    print("Saved test_sentences_6k.txt for predictor held-out evaluation.")

if __name__ == "__main__":
    main()
