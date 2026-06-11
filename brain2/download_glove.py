import gensim.downloader as api
import struct
import os

print("Downloading GloVe 50-dimensional embeddings (approx 65MB)...")
model = api.load("glove-wiki-gigaword-50")

# We want 128 dimensions to match the brain's internal architecture
# We will pad the 50d vectors with zeros to reach 128d
TARGET_DIM = 128
vocab_size = len(model.key_to_index)
print(f"Downloaded vocab size: {vocab_size}")

out_file = "checkpoints/semantic_dict.bin"
os.makedirs("checkpoints", exist_ok=True)

print(f"Saving to {out_file}...")
with open(out_file, "wb") as f:
    # Write metadata: vocab_size (int32), dim (int32)
    f.write(struct.pack("ii", vocab_size, TARGET_DIM))
    
    for word, idx in model.key_to_index.items():
        # Encode word as null-terminated string
        word_bytes = word.encode('utf-8', errors='ignore')
        if len(word_bytes) > 63:
            word_bytes = word_bytes[:63] # Truncate long words
        f.write(word_bytes + b'\x00')
        
        # Get 50d vector
        vec = model.vectors[idx]
        
        # Write first 50 dims
        for val in vec:
            f.write(struct.pack("f", float(val)))
            
        # Pad remaining 78 dims with 0
        for _ in range(TARGET_DIM - len(vec)):
            f.write(struct.pack("f", 0.0))

print("Done! Semantic embedding map created successfully.")
