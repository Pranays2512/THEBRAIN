import brain2
import time

print("Initializing Sparse MoE Brain...")
b = brain2.Brain(som_rows=512, som_cols=512, n_dims=512, hidden_dim=512)

import os
import numpy as np

glove_txt = "glove.6B.50d.txt"
if os.path.exists(glove_txt):
    print("Loading GloVe Semantic Embeddings...")
    with open(glove_txt, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            if not word.isalpha(): continue
            vec50 = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            vec512 = np.zeros(512, dtype=np.float32)
            vec512[:50] = vec50
            b.language.register_word(word, vec512)
    b.language.freeze_vocabulary()
    print("Vocabulary frozen with semantic clusters!")

# Tiny dataset to test convergence of the Sparse LSTM
dataset = [
    ("Hello Brain", "Hi human!"),
    ("What is your name?", "I am Antigravity."),
    ("Are you fast?", "Yes, I am a sparse tree."),
]

print("\nTraining Sparse LSTM...")
epochs = 50

start_time = time.time()
for epoch in range(epochs):
    epoch_error = 0
    for input_text, target_text in dataset:
        b.perceive_text(input_text)
        
        # We manually encode the target to pass to train_sequence
        target_words = target_text.lower().replace("?", " ?").replace("!", " !").replace(",", " ,").replace(".", " .").split()
        inputs = []
        for w in target_words:
            if not b.language.knows(w):
                b.language.register_word(w)
            inputs.append(b.language.encode(w))
        
        if len(inputs) > 0:
            target_vec = inputs[-1] # Train to predict the last word
            # Predictor train sequence expects input history, and target vector
            error = b.predictor.train_sequence(inputs[:-1] if len(inputs)>1 else inputs, target_vec, -1)
            epoch_error += error
            
        b.reset_sequence()
        
    avg_error = epoch_error / len(dataset)
    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch}: L2 Prediction Error = {avg_error:.4f}")

end_time = time.time()
print(f"\nTraining completed in {end_time - start_time:.4f} seconds!")

print("\n--- Testing Generation ---")
for input_text, _ in dataset:
    print(f"\nUser: {input_text}")
    b.perceive_text(input_text)
    res = b.think(5)
    clean_response = [w for w in res.words if w and w != "<pad>" and w != "<unk>"]
    print(f"Brain: {' '.join(clean_response)}")
    b.reset_sequence()
