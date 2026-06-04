import os
import random

def generate():
    cases = []
    
    # 1. Semantic Query (100)
    for i in range(100):
        cases.append(f"semantic | bird{i} isa animal{i} | bird{i} isa ? | animal{i}")
        
    # 2. Describe (100)
    for i in range(100):
        cases.append(f"describe | car{i} has wheels{i} ; car{i} has doors{i} | describe car{i} | car{i} has wheels{i}. car{i} has doors{i}.")

    # 3. Algebra (100)
    for i in range(100):
        a = random.randint(1, 10)
        b = random.randint(0, 50)
        c = random.randint(10, 100)
        ans = float((c - b) // a)
        cases.append(f"algebra | | {a} x + {b} = {c} | x = {ans:.2f}")

    # 4. Permute (100)
    for i in range(100):
        n = random.randint(3, 7)
        k = random.randint(1, n)
        # N! / (N-K)!
        import math
        ans = math.perm(n, k)
        cases.append(f"permute | | {n} permute {k} | {ans}")

    # 5. Probability (100)
    for i in range(100):
        n = random.randint(1, 50)
        d = random.randint(n, 100)
        # Use round-half-away-from-zero to match C++ snprintf %.2f
        import math as _math
        raw = n / d
        ans = _math.floor(raw * 100 + 0.5) / 100
        cases.append(f"probability | | probability of {n} out of {d} | {ans:.2f}")

    # 6. Area (100)
    for i in range(100):
        w = random.randint(1, 100)
        h = random.randint(1, 100)
        cases.append(f"area | | area of {w} and {h} | {w*h}")

    # 7. Power (100)
    for i in range(100):
        b = random.randint(1, 10)
        p = random.randint(0, 4)
        cases.append(f"power | | {b} power {p} | {b**p}")

    # 8. Grammar (100)
    for i in range(100):
        cases.append(f"grammar | apple{i} isa fruit{i} | what isa apple{i} | fruit{i}")

    # 9. Self-Ontology (100)
    for i in range(100):
        cases.append(f"self | pranay{i} is creator{i} | who is pranay{i} | creator{i}")

    # 10. Episodic Memory (100)
    for i in range(100):
        cases.append(f"episodic | obj{i} | obj{i} | obj{i}")

    # 11. Causal Reasoning (100)
    for i in range(100):
        cases.append(f"causal | nodeA{i} causes nodeB{i} ; nodeB{i} causes nodeC{i} ; nodeC{i} causes nodeD{i} | nodeA{i} causes ? | nodeD{i}")

    with open("tests/test_hardened_1100.txt", "w") as f:
        for c in cases:
            f.write(c + "\n")

if __name__ == "__main__":
    generate()
