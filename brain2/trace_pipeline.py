#!/usr/bin/env python3
import sys, os, time
import json
import brain2

def get_cpp_memory_usage():
    # Since we can't easily interrogate the C++ heap from Python, we statically compute 
    # the exact memory allocated by the architecture based on the dimensionalities.
    N_DIMS = 128
    SOM_NEURONS = 256 * 256  # 65,536

    storage = {
        "SOM_Weights": (SOM_NEURONS * N_DIMS * 4) / (1024*1024),
        "WorkingMemory_Buffer": (SOM_NEURONS * 7 * 4) / (1024*1024),
        "EpisodicMemory_Nodes": (2000 * 50 * 2) / (1024*1024), # Sparse bit arrays (avg)
        "Language_Dictionary": (400000 * N_DIMS * 4) / (1024*1024),
        "Predictor_MDN": (N_DIMS * 256 * 4 + 256 * N_DIMS * 5 * 4) / (1024*1024),
        "BasalGanglia_MLP": (N_DIMS * 256 * 4 + 256 * 32 * 4) / (1024*1024),
        "BindingMemory_LSH": (2000 * N_DIMS * 3 * 4) / (1024*1024),
        "ProceduralMemory_Triggers": (200 * N_DIMS * 4) / (1024*1024),
        "PredictiveCoding_SOM": (SOM_NEURONS * 4) / (1024*1024),
        "GlobalWorkspace": (SOM_NEURONS * 4) / (1024*1024),
    }
    return storage

def trace_input(b, text_input):
    print(f"\n=============================================")
    print(f"TRACING INPUT: '{text_input}'")
    print(f"=============================================")
    
    # 1. Clear previous profile times
    # In brain2 we'll just run it and grab the accumulated string
    
    # 2. Run the trace
    t0 = time.time()
    if any(op in text_input for op in ["+", "-", "*", "/", "eval", "roots"]):
        print("-> Detected Math Equation. Routing to Reason() / LogicEngine")
        # For math, we push to scratchpad and call direct_reason_step
        tokens = text_input.split()
        if len(tokens) >= 3:
            b.scratchpad.write("subject", b.language.encode(tokens[0]), "math_arg")
            b.scratchpad.write("object", b.language.encode(tokens[2]), "math_arg")
        if len(tokens) >= 2:
            b.scratchpad.write("a_operator", b.language.encode(tokens[1]), "math_arg")
            
        op_picked = b.direct_reason_step("reply")
        reply = "Math Operation Executed: " + str(op_picked)
    else:
        print("-> Detected Natural Language. Routing to Cognitive_Step (Perceive -> Think -> Speak)")
        reply = b.cognitive_step(text_input)
    t1 = time.time()
    
    print(f"\n--- Output ---")
    print(f"System Reply: {reply}")
    print(f"Total Python Latency: {(t1 - t0) * 1000:.2f} ms")
    
    print(f"\n--- C++ Microsecond Profiling (Module Bottlenecks) ---")
    print(b.get_profiling_report())
    
def main():
    print("Initializing Brain2 Engine for Tracing...")
    t0 = time.time()
    b = brain2.Brain(som_rows=256, som_cols=256, n_dims=128, hidden_dim=256)
    
    ckpt_dir = "checkpoints/math_brain"
    if os.path.exists(f"{ckpt_dir}/predictor.bin"):
        b.load_components(
            predictor_path=f"{ckpt_dir}/predictor.bin",
            language_path=f"{ckpt_dir}/language.bin",
            som_path=f"{ckpt_dir}/som.bin",
            episodic_path=f"{ckpt_dir}/episodic.bin",
            emotion_path=f"{ckpt_dir}/emotion.bin",
            self_path=f"{ckpt_dir}/self.bin",
            symbolic_path=f"{ckpt_dir}/symbolic.bin",
            binding_path=f"{ckpt_dir}/binding.bin",
            bg_path=f"{ckpt_dir}/bg.bin",
            procedures_path=f"{ckpt_dir}/procedures.bin",
            hpred_path=f"{ckpt_dir}/hpred.bin"
        )
    
    if os.path.exists("checkpoints/semantic_dict.bin"):
        b.language.load_semantics("checkpoints/semantic_dict.bin")
        
    print(f"Brain boot time: {(time.time() - t0):.2f}s")
    
    print(f"\n--- Storage Profile (Memory Footprint) ---")
    storage = get_cpp_memory_usage()
    total = 0
    for mod, sz in storage.items():
        print(f"{mod:25s}: {sz:>8.2f} MB")
        total += sz
    print(f"-"*36)
    print(f"TOTAL ESTIMATED C++ HEAP  : {total:>8.2f} MB\n")

    # Trace a word/sentence
    trace_input(b, "hello how are you")
    
    # Trace a math equation
    trace_input(b, "45 + 12")

if __name__ == "__main__":
    main()
