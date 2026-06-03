import os
import sys
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

# Ensure brain2 module can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
try:
    import brain2
except ImportError as e:
    print(f"Error importing brain2: {e}")
    sys.exit(1)

app = Flask(__name__, static_url_path='', static_folder='static')

# Ensure deterministic RNG state by pre-registering all words and symbols alphabetically
def initialize_embeddings(b):
    b.symbolic_table.seed_math_symbols()
    for i in range(1000):
        b.symbolic_table.bind(str(i))
    words = sorted(["x", "goal", "relation", "math", "op_symbol", "object", "comparison", "eval", "subject", "result"])
    for word in words:
        b.language.register_word(word)
        b.symbolic_table.bind(word)

print("Initializing Brain...")
# Note: visualizer_api used to use n_dims=64! We MUST use 16 to match the trained model!
b = brain2.Brain(som_rows=10, som_cols=10, n_dims=16, episodic_max=10000)
initialize_embeddings(b)



checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", "bg_ep200000.bin")
if os.path.exists(checkpoint_path):
    print(f"Loading trained weights from {checkpoint_path}")
    b.load_bg(checkpoint_path)
else:
    print(f"WARNING: No trained weights found at {checkpoint_path}")

# Helper to encode scalars (copied from train_algebra.py)
def encode_scalar(brain, value):
    sym = str(value)
    sym = sym.rstrip('0').rstrip('.') if '.' in sym else sym
    if not brain.symbolic_table.knows(sym):
        brain.symbolic_table.bind(sym)
    return brain.symbolic_table.lookup(sym)

def setup_equation(a, b_val, c):
    b.scratchpad.clear()
    b.scratchpad.write("subject",  np.array(encode_scalar(b, c),  dtype=np.float32), "math")
    b.scratchpad.write("object",   np.array(encode_scalar(b, b_val),  dtype=np.float32), "math")
    b.scratchpad.write("relation", np.array(encode_scalar(b, a),  dtype=np.float32), "math")
    goal_vec = np.array(b.language.encode("x"), dtype=np.float32)
    b.scratchpad.write("goal",     goal_vec, "goal")
    b.scratchpad.write("op_symbol", np.array(encode_scalar(b, 0), dtype=np.float32), "-")
    b.scratchpad.write("comparison", np.zeros(b.scratchpad.n_dims, dtype=np.float32), "eval")
    
    # We must call start_reasoning here so the tree is ready and traces are cleared
    b.start_reasoning()

OP_NAMES = ["READ", "WRITE", "MATH_SUB", "MATH_DIV", "COMPARE", "BIND_QUERY", "RETRIEVE", "ANALOGY", "HALT"]

@app.route("/")
def index():
    return send_from_directory('static', 'index.html')

@app.route("/api/problem", methods=["POST"])
def new_problem():
    data = request.json
    a = data.get("a", 2)
    b_val = data.get("b", 4)
    c = data.get("c", 10)
    
    setup_equation(a, b_val, c)
    return jsonify({"status": "ok", "equation": f"{a}x + {b_val} = {c}"})

@app.route("/api/step", methods=["POST"])
def step():
    op_idx = b.reason_step("x", 0.0) # Greedy lookahead, no epsilon
    op_name = OP_NAMES[op_idx] if 0 <= op_idx < len(OP_NAMES) else "UNKNOWN"
    return jsonify({"status": "ok", "picked_op": op_name})

@app.route("/api/tree", methods=["GET"])
def get_tree():
    tree = b.scratchpad.get_tree()
    current_node = b.scratchpad.current_node()
    
    nodes_data = []
    for node in tree:
        nodes_data.append({
            "id": node.id,
            "parent_id": node.parent_id,
            "h_cost": float(node.h_cost),
            "is_current": (node.id == current_node)
        })
        
    return jsonify({
        "nodes": nodes_data,
        "current_node": current_node
    })

@app.route("/api/scratchpad", methods=["GET"])
def get_scratchpad():
    slots = b.scratchpad.slot_names()
    data = {}
    for slot in slots:
        vec = b.scratchpad.read(slot)
        if len(vec) > 0:
            tag = b.scratchpad.tag(slot)
            if tag == "goal":
                val_str = b.language.best_word(vec)
            else:
                val_str = b.symbolic_table.nearest_symbol(vec)
                if not val_str: # fallback
                    val_str = b.language.best_word(vec)
        else:
            val_str = "empty"
            
        data[slot] = {
            "tag": b.scratchpad.tag(slot),
            "symbol": val_str
        }
    return jsonify(data)

if __name__ == "__main__":
    app.run(port=5001, debug=True)
