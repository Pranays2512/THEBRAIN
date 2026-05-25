import os
import sys
import numpy as np

# Ensure brain2 module can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    import brain2
except ImportError:
    print("Error: brain2 module not found. Did you run ./build.sh?")
    sys.exit(1)

def test_structure_mapping():
    # Initialize a Brain instance
    b = brain2.Brain(som_rows=4, som_cols=4, n_dims=16)
    
    # Random embeddings for our entities and relations
    fire = np.random.randn(16).astype(np.float32)
    oxygen = np.random.randn(16).astype(np.float32)
    burn = np.random.randn(16).astype(np.float32)
    
    electricity = np.random.randn(16).astype(np.float32)
    water = np.random.randn(16).astype(np.float32)
    shock = np.random.randn(16).astype(np.float32)
    
    # Let's say oxygen and water are both "elements" - give them a slightly similar vector
    water = oxygen * 0.5 + np.random.randn(16).astype(np.float32) * 0.5
    
    # We define relations for "combines_with" and "causes"
    combines_with = np.random.randn(16).astype(np.float32)
    causes = np.random.randn(16).astype(np.float32)
    
    # Bind: fire combines_with oxygen -> burn
    b.bind_triple(fire, combines_with, burn)
    
    # Bind: electricity combines_with water -> shock
    b.bind_triple(electricity, combines_with, shock)
    
    # Test normal query
    vec, conf = b.binding_query(electricity, combines_with, want_object=True, threshold=0.3)
    assert conf > 0.3, "Failed to retrieve direct binding"
    
    # Now simulate an analogy: We know 'fire combines_with oxygen'. What happens with 'electricity combines_with water'?
    # We query the analogy engine with 'electricity' and 'combines_with', but we also have 'water' in the context.
    # To put 'water' in context, we perceive it so it goes into working memory.
    b.perceive(water)
    b.perceive(electricity)
    
    # Analogy structure map: (electricity, combines_with) -> ?
    analogy_result = b.analogy_op(electricity, combines_with)
    
    # Check if analogy_result is close to 'shock'
    sim = np.dot(analogy_result, shock) / (np.linalg.norm(analogy_result) * np.linalg.norm(shock) + 1e-9)
    print(f"Analogy similarity to 'shock': {sim:.3f}")
    assert sim > 0.5, "Analogy engine failed to map structural relation"

if __name__ == '__main__':
    test_structure_mapping()
    print("test_analogy.py passed.")
