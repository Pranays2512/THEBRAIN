import brain2
b = brain2.Brain(8, 8, 16)
ckpt = "checkpoints/stage5_math"
b.load_components(
    predictor_path=f"{ckpt}/predictor.bin",
    language_path=f"{ckpt}/language.bin",
    som_path=f"{ckpt}/som.bin",
    episodic_path=f"{ckpt}/episodic.bin"
)

# Simulate sentences
def sentence(words_str, subj_str):
    for w in words_str.split():
        v = b.language.encode(w)
        b.perceive(v)
    subj_v = b.language.encode(subj_str)
    b.commit_episode(1.0, subj_v)
    print(f"Committed '{words_str}' with payload '{subj_str}'. Total: {b.episodic.episode_count()}")

sentence("i am holding an apple", "i")
sentence("apple is red", "apple")

# Query
v_red = b.language.encode("red")
focus = b.som.activation_map(v_red)
topk = b.episodic.retrieve_topk(focus, 2)
print("Topk for 'red':", topk)
for sim, idx in topk:
    print(f"Sim {sim} -> idx {idx} payload {b.language.best_word(b.episodic.get_last_episode())}") # Just approximate
