from tests.run_hardened_suite import load_brain
import brain2

b = load_brain()

words = "nodeA4 causes nodeB4 ; nodeB4 causes nodeC4 ; nodeC4 causes nodeD4".replace(";", "").split()
for w in words: b.language.register_word(w)

for i in range(0, len(words), 3):
    subj, rel, obj = words[i], words[i+1], words[i+2]
    b.binding.bind(b.language.encode(subj), b.language.encode(rel), b.language.encode(obj))

qwords = "nodeA4 causes ?".split()
for w in qwords: b.language.register_word(w)

def check_q(subj):
    vec, conf = b.binding.query(b.language.encode(subj), b.language.encode("causes"), True, 0.3, 4)
    print(f"Query {subj} causes ? -> {b.language.best_word(vec)} {conf}")

check_q("nodeC4")
check_q("nodeB4")
check_q("nodeA4")

