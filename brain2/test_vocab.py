import chat
b = chat.brain2.Brain(som_rows=8, som_cols=8, n_dims=16)
b.learn_word("isa")
b.learn_word("reply")

words = ["blarg", "isa", "florp"]
print("Before:", [b.symbolic_table.knows(w) for w in words])

for w in words:
    if not b.symbolic_table.knows(w):
        b.learn_word(w)
        print(f"Learned {w}")
    vec = b.language.encode(w)
    print(f"{w} vector len: {len(vec)}, sum_abs: {sum(abs(x) for x in vec)}")

print("After:", [b.symbolic_table.knows(w) for w in words])
