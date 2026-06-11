import brain2
import os

print("Loading 512D Graph Architecture...")
# The checkpoints have been saved from the massive training.
b = brain2.Brain(som_rows=512, som_cols=512, n_dims=512, hidden_dim=512)

# Load the trained checkpoints
ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "executive_brain")
if os.path.exists(ckpt_dir):
    print(f"Loading checkpoints from {ckpt_dir}...")
    try:
        b.load_components(
            predictor_path=os.path.join(ckpt_dir, "predictor.bin"),
            language_path=os.path.join(ckpt_dir, "language.bin"),
            som_path=os.path.join(ckpt_dir, "som.bin"),
            episodic_path=os.path.join(ckpt_dir, "episodic.bin"),
            emotion_path=os.path.join(ckpt_dir, "emotion.bin"),
            self_path=os.path.join(ckpt_dir, "self.bin"),
            symbolic_path=os.path.join(ckpt_dir, "symbolic.bin"),
            binding_path=os.path.join(ckpt_dir, "binding.bin"),
            bg_path=os.path.join(ckpt_dir, "bg.bin"),
            procedures_path=os.path.join(ckpt_dir, "procedures.bin"),
            hpred_path=os.path.join(ckpt_dir, "hpred.bin")
        )
    except Exception as e:
        print(f"Failed to load: {e}")
else:
    print("WARNING: Checkpoints folder not found!")

# Turn on speaking mode for testing

prompts = [
    "Hello! How are you doing today?",
    "What is the capital of France?",
    "Who is the president of the United States?",
    "Can you tell me about the architecture of a computer?",
    "What is your favorite thing to think about?"
]

print("\n--- Fluency & Fact Evaluation ---\n")

for p in prompts:
    print(f"\nUser: {p}")
    # Process the entire prompt
    b.perceive_text(p)
        
    print("Brain is thinking...")
    # Generate 15 words of response
    res = b.think(15)
    
    # Filter out empty strings or padding from the words list
    clean_response = [w for w in res.words if w and w != "<pad>" and w != "<unk>"]
    
    print(f"Brain: {' '.join(clean_response)}")
    b.reset_sequence()
