import brain2
b = brain2.Brain(n_dims=128, som_rows=1, som_cols=1, hidden_dim=256)
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
print("Episodic size:", b.episodic.size)
print("Symbolic size:", b.symbolic.size)
print("Binding size:", b.binding.size)
