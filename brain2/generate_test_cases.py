import random

features = {
    "semantic_query": [],
    "describe": [],
    "algebra": [],
    "permute": [],
    "probability": [],
    "area": [],
    "power": [],
}

animals = ["dog", "cat", "bird", "fish", "elephant", "lion", "tiger", "bear", "fox", "wolf"]
attributes = ["animal", "mammal", "pet", "creature", "predator", "prey"]
relations = ["isa", "has", "can"]

# 1. Semantic Query (100 cases)
for i in range(100):
    subj = random.choice(animals)
    rel = random.choice(relations)
    features["semantic_query"].append(f"{subj} {rel} ?")

# 2. Describe (100 cases)
for i in range(100):
    subj = random.choice(animals + ["sun", "earth", "moon", "apple", "pizza", "car", "robot"])
    features["describe"].append(f"describe {subj}")

# 3. Basic Algebra (100 cases: a * x + b = c  =>  x = (c - b) / a)
# Ensure x is an integer
for i in range(100):
    a = random.randint(1, 15)
    x = random.randint(1, 20)
    b = random.randint(1, 50)
    c = a * x + b
    features["algebra"].append(f"{a} x + {b} = {c}")

# 4. Math: Permutations (100 cases: n permute r)
for i in range(100):
    n = random.randint(3, 8)
    r = random.randint(1, n-1)
    features["permute"].append(f"{n} permute {r}")

# 5. Math: Probability (100 cases: target / total)
# Ensure clean division for test grading simplicity (or just random)
for i in range(100):
    target = random.randint(1, 20)
    total = target * random.randint(1, 10)
    features["probability"].append(f"probability of {target} in {total}")

# 6. Math: Area (100 cases: area of l and b)
for i in range(100):
    l = random.randint(1, 50)
    w = random.randint(1, 50)
    features["area"].append(f"area of {l} and {w}")

# 7. Math: Power (100 cases: base power exp)
for i in range(100):
    base = random.randint(1, 10)
    exp = random.randint(1, 4)
    features["power"].append(f"{base} power {exp}")


# Write all to a file with labels
with open("test_700.txt", "w") as f:
    for feature_name, cases in features.items():
        for case in cases:
            f.write(f"[{feature_name}] {case}\n")
    
print(f"Generated {sum(len(v) for v in features.values())} test cases across {len(features)} features.")
