# brain2/_refactor/classify.py
"""Assign every top-level brain2 module to a layer/subpackage from real imports +
name heuristics. Writes move_map.json {old_module: 'pkg/sub'} and prints groups +
up-pointing edges for human review. Pure analysis — moves nothing."""
import ast, json, os, re, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # brain2/

def modules():
    out = {}
    for f in os.listdir(ROOT):
        if f.endswith(".py") and f != "__init__.py":
            p = os.path.join(ROOT, f)
            try: tree = ast.parse(open(p, errors="ignore").read())
            except Exception: continue
            deps = set()
            for n in ast.walk(tree):
                if isinstance(n, ast.Import):
                    for a in n.names: deps.add(a.name.split(".")[0])
                elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                    deps.add(n.module.split(".")[0])
            out[f[:-3]] = deps
    return out

TEST = re.compile(r"^(test_|harden_|stress_|.*_exam$)|^(validate|component_validation|reasoning_suite|exam|exam_math)$")
TRAIN = re.compile(r"^(train_|read_pdf_train$|auto_|knowledge_distill$|student_)")
IO = {"llm_adapter","llm_extractor","mouth","ocr_pdf","nl_front","server","chat",
      "brain_repl","structural_parser","math_parser","integrated_front","converse"}
FAC = {"whole_brain","read_book","reading_loop","conversation_engine","query_planner",
       "neuro_bridge","event_predict","feature_learner"}
CORE_SUB = [
 ("synthesis", re.compile(r"synth|proposer|program_synth|loop_synth|dp_|composable|refut|invariant|conjecture|code_gen|codegen")),
 ("math", re.compile(r"algebra|calculus|integral|physics|word_math|factoriz|dimensional|prob_compute|math_")),
 ("grounding", re.compile(r"ground|grounding|crispify|context_embed|domain_features")),
 ("events", re.compile(r"event_|verb_|discourse|analogy|compositional")),
 ("knowledge", re.compile(r"knowledge|concept|semantic|world_knowledge|core_knowledge|fact_extractor|conceptnet|taxonom")),
 ("neural", re.compile(r"neural_lm|cpp_accel")),
 ("store", re.compile(r"brain_store|check_library|template_memory|type_oracle|parse_template|corpus_scale|coverage_harness")),
 ("reasoning", re.compile(r"reason|tree_|means_ends|nested_parser|deeper_grammar|dual_process|planning|planner|learned_guidance")),
]

def layer_of(m):
    if TEST.match(m): return "tests"
    if TRAIN.match(m): return "training"
    if m in IO: return "io"
    if m in FAC: return "faculties"
    for sub, rx in CORE_SUB:
        if rx.search(m): return f"core/{sub}"
    return "core/misc"

def main():
    mods = modules()
    mp = {m: layer_of(m) for m in mods}
    groups = collections.defaultdict(list)
    for m, dst in mp.items(): groups[dst].append(m)
    for dst in sorted(groups):
        print(f"\n{dst} ({len(groups[dst])}): " + ", ".join(sorted(groups[dst])))
    rank = lambda d: {"io":1,"faculties":2,"training":3,"tests":4}.get(d.split("/")[0], 0)
    local = set(mods); ups = []
    for m, ds in mods.items():
        for d in ds & local - {m}:
            if rank(mp[d]) > rank(mp[m]): ups.append((m, mp[m], d, mp[d]))
    print(f"\nUP-EDGES to review ({len(ups)}):")
    for m, lm, d, ld in ups: print(f"  {m}({lm}) -> {d}({ld})")
    json.dump(mp, open(os.path.join(os.path.dirname(__file__), "move_map.json"), "w"), indent=1)
    print(f"\nwrote move_map.json ({len(mp)} modules)")

if __name__ == "__main__":
    main()
