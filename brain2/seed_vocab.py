import brain2

def seed_vocabulary():
    print("Initializing Brain...")
    b = brain2.Brain(som_rows=256, som_cols=256, n_dims=128, hidden_dim=256)
    
    # Load current weights so we don't overwrite them
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
    
    print("Registering numbers -100 to 200...")
    for i in range(-100, 201):
        num_str = str(i)
        if not b.language.knows(num_str):
            b.language.register_word(num_str)
            # Also bind it symbolically just in case it's used as a slot value
            if not b.symbolic_table.knows(num_str):
                b.symbolic_table.bind(num_str)

    print("Registering math symbols and tokens...")
    symbols = ["+", "-", "*", "/", "^", "roots", "of", "eval", "for", "is", "are", "and", "=", "x"]
    for sym in symbols:
        if not b.language.knows(sym):
            b.language.register_word(sym)
            if not b.symbolic_table.knows(sym):
                b.symbolic_table.bind(sym)
                
    # Also seed the mathematical structural slots to guarantee they are mapped
    slots = ["SLOT_a_operator", "SLOT_subject", "SLOT_object", "SLOT_relation", "SLOT_result", "SLOT_focus", "SLOT__context"]
    for s in slots:
        if not b.symbolic_table.knows(s):
            b.symbolic_table.bind(s)
                
    # Save the updated language and symbolic dictionaries
    print("Saving updated components...")
    b.save_components("checkpoints/math_brain")
    print("Done!")

if __name__ == "__main__":
    seed_vocabulary()
