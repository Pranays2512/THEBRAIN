import brain2
import os

b = brain2.Brain(8, 8, 16)
checkpoint_dir = "checkpoints/stage5_math"
b.load_components(
    predictor_path=os.path.join(checkpoint_dir, "predictor.bin"),
    language_path=os.path.join(checkpoint_dir, "language.bin"),
    som_path=os.path.join(checkpoint_dir, "som.bin"),
    episodic_path=os.path.join(checkpoint_dir, "episodic.bin"),
    emotion_path=os.path.join(checkpoint_dir, "emotion.bin"),
    self_path=os.path.join(checkpoint_dir, "self.bin"),
    symbolic_path=os.path.join(checkpoint_dir, "symbolic.bin"),
    binding_path=os.path.join(checkpoint_dir, "binding.bin"),
    bg_path=os.path.join(checkpoint_dir, "bg.bin"),
    procedures_path=os.path.join(checkpoint_dir, "procedures.bin"),
    hpred_path=os.path.join(checkpoint_dir, "hpred.bin")
)
b.symbolic_table.seed_math_symbols()
for i in range(1000):
    b.symbolic_table.bind(str(i))
    
def test_seq(subj, obj, goal):
    b.reset_sequence()
    b.scratchpad.write("subject", b.language.encode(subj), "context")
    b.scratchpad.write("object", b.language.encode(obj), "context")
    print("Subj best word:", b.language.best_word(b.scratchpad.read("subject")))
    print("Obj best word:", b.language.best_word(b.scratchpad.read("object")))
    print("sum of 20:", sum(abs(x) for x in b.language.encode("20")))
    goal_vec = b.language.encode(goal)
    print("Goal sum:", sum(abs(x) for x in goal_vec))
    b.scratchpad.write("goal", goal_vec, "goal")
    bmu = b.som.activation_map(goal_vec)
    gated = b.working_mem.gate(bmu * 10.0, 1.0)
    print("Gated:", gated)
    b.working_mem.tick()
    
    ctx = b.working_mem.context()
    print("Ctx sum:", sum(abs(x) for x in ctx))
    
    seq = b.procedures.retrieve(ctx)
    print("Retrieved seq:", seq)
    for op in seq:
        b.force_reason_step(op, goal)
    res_vec = b.scratchpad.read("result")
    print("Result read vec:", res_vec)
    print("Result read sum:", sum(abs(x) for x in res_vec))
    spoken = b.get_spoken_words()
    b.clear_spoken_words()
    print(f"Result for {goal} ({subj}, {obj}):", spoken)

test_seq("5", "2", "permute")
test_seq("5", "4", "area")
test_seq("5", "2", "power")
