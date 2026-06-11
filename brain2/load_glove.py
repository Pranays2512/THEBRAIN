import os
import urllib.request
import zipfile
import numpy as np
import brain2

# 1. Download GloVe
glove_zip = "glove.6B.zip"
glove_txt = "glove.6B.50d.txt"
glove_url = "http://nlp.stanford.edu/data/glove.6B.zip"

if not os.path.exists(glove_txt):
    if not os.path.exists(glove_zip):
        print("Downloading GloVe embeddings (this might take a few minutes)...")
        urllib.request.urlretrieve(glove_url, glove_zip)
    print("Extracting GloVe...")
    with zipfile.ZipFile(glove_zip, 'r') as zip_ref:
        zip_ref.extract(glove_txt)

# 2. Inject into Brain
print("Initializing 512D Brain Architecture...")
b = brain2.Brain(som_rows=512, som_cols=512, n_dims=512, hidden_dim=512)

print("Injecting GloVe 50D into Brain's 512D Language Module...")
count = 0
with open(glove_txt, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.split()
        word = parts[0]
        # Only inject standard alphabetical words to keep vocabulary clean
        if not word.isalpha(): continue
        
        vec50 = np.array([float(x) for x in parts[1:]], dtype=np.float32)
        # Pad to 512D
        vec512 = np.zeros(512, dtype=np.float32)
        vec512[:50] = vec50
        
        # Inject into Brain language module
        b.language.register_word(word, vec512)
        count += 1
        if count % 50000 == 0:
            print(f"Loaded {count} words...")

print(f"Successfully injected {count} meaningful word vectors into the Brain!")

# Freeze the vocabulary to prevent semantic drift!
b.language.freeze_vocabulary()

ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "executive_brain")
os.makedirs(ckpt_dir, exist_ok=True)
lang_path = os.path.join(ckpt_dir, "language.bin")
b.language.save(lang_path)
print(f"Saved frozen vocabulary with semantic clusters to {lang_path}")
