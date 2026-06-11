import brain2
import json
import cProfile, pstats

def main():
    b = brain2.Brain(som_rows=256, som_cols=256, n_dims=128, hidden_dim=256)
    b.load_components(
        predictor_path="checkpoints/math_brain/predictor.bin",
        language_path="checkpoints/math_brain/language.bin",
        som_path="checkpoints/math_brain/som.bin",
        episodic_path="checkpoints/math_brain/episodic.bin",
        emotion_path="checkpoints/math_brain/emotion.bin",
        self_path="checkpoints/math_brain/self.bin",
        symbolic_path="checkpoints/math_brain/symbolic.bin",
        binding_path="checkpoints/math_brain/binding.bin",
        bg_path="checkpoints/math_brain/bg.bin",
        procedures_path="checkpoints/math_brain/procedures.bin",
        hpred_path="checkpoints/math_brain/hpred.bin"
    )

    with open("data/math_corpus.json") as f:
        math_corpus = json.load(f)

    for pair in math_corpus[:50]:
        b.reset_sequence()
        inp = pair["input"]
        tokens = inp.split()
        
        if inp.startswith("eval"):
            b.scratchpad.write("subject", b.language.encode(tokens[13]), "math_arg")
            b.scratchpad.write("object", b.language.encode(tokens[1]), "math_arg")
            b.scratchpad.write("a_operator", b.language.encode(tokens[5] + tokens[6]), "math_arg")
            b.scratchpad.write("focus", b.language.encode(tokens[8] + tokens[9]), "math_arg")
        else:
            if len(tokens) >= 3:
                b.scratchpad.write("subject", b.language.encode(tokens[0]), "math_arg")
                b.scratchpad.write("object", b.language.encode(tokens[2]), "math_arg")
            if len(tokens) >= 2:
                b.scratchpad.write("a_operator", b.language.encode(tokens[1]), "math_arg")
                
        b.start_reasoning()
        b.force_reason_step(20, "reply")
        b.reinforce_bg(1.0)

cProfile.run('main()', 'prof_stats')
p = pstats.Stats('prof_stats')
p.sort_stats('cumtime').print_stats(30)
