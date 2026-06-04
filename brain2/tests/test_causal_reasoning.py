import brain2

def test_causal_reasoning():
    b = brain2.Brain(8, 8, 16)
    
    words = ["A", "B", "C", "D", "E", "bigger"]
    for w in words:
        b.language.register_word(w)
        b.symbolic_table.bind(w)
        
    vA = b.language.encode("A")
    vB = b.language.encode("B")
    vC = b.language.encode("C")
    vD = b.language.encode("D")
    vE = b.language.encode("E")
    rel = b.language.encode("bigger")
    
    b.binding.bind(vA, rel, vB)
    b.binding.bind(vB, rel, vC)
    b.binding.bind(vC, rel, vD)
    b.binding.bind(vD, rel, vE)
    
    print("Testing query: A > ? (depth 4)")
    ans_vec, conf = b.binding.query(vA, rel, True, 0.3, 4)
    ans_word = b.language.best_word(ans_vec)
    print(f"Result: {ans_word} (conf: {conf:.3f})")
    assert ans_word == "E", f"Expected E, got {ans_word}"
    
    print("Testing B > ? (depth 3)")
    ans_vec, conf = b.binding.query(vB, rel, True, 0.3, 3)
    ans_word = b.language.best_word(ans_vec)
    print(f"Result: {ans_word} (conf: {conf:.3f})")
    assert ans_word == "E", f"Expected E, got {ans_word}"

    print("All causal reasoning tests passed!")

if __name__ == "__main__":
    test_causal_reasoning()
