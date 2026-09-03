# THEBRAIN

Neurosymbolic cognitive architecture. C++17/20, header-only core, no LLM in the runtime path.

## Layout — read this first

| Dir | Status |
|---|---|
| `brain3/` | **the project.** All work happens here |
| `brain2/` | **legacy.** Kept only as a corpus source — brain3 ingests `brain2/data/*.txt` |
| `Frontend/` | user-built UI. **Do not modify.** Standing directive |
| `obsidian_vault/`, `graphify-out/` | generated / notes, not source |

## Architecture in one screen

Bicameral. Two hemispheres, one orchestrator, one query language.

- **`brain3/fuzzy/`** (~11k lines) — continuous, learns. SOM graph, sparse LSTM predictor,
  predictive coding, working memory, episodic store, emotion, TD-λ basal ganglia. ~1M params.
- **`brain3/crisp/`** (~30k lines) — exact, verifies. Math/algebra/calculus/integral, causal
  SCM (Pearl L1–L3), analogy (Gentner SME), discovery (BACON), metacognitive refuter,
  code synth, instinct reflexes.
- **`brain3/core/master_orchestrator.hpp`** (1855 lines) — the actual runtime.

Everything speaks **BrainQL** internally (LOOKUP / CHAIN / INHERIT / SOLVE / SYNTH / COMPUTE /
REFUTE / INTERVENE / COUNTERFACTUAL / DISCOVER / ANALOGY / INSTINCT / INGEST). Natural language
is converted at the boundary and never reaches the engines.

**Core design invariant:** the fuzzy→crisp membrane opens only on `verified == true`. The neural
side can never write into the truth store. This is the anti-hallucination guarantee — preserve it
in any change.

### Live turn path — `process()` → `process_core()`

| Stage | What | Where |
|---|---|---|
| 0 | `_fuzzy_pass` — LM trains on the raw utterance *before* symbolic resolution | `:1406` |
| 1 | `parse_intent_to_bql` — NL → BrainQL, native C++, no LLM | `:617` |
| 2 | crisp executor answers; `_consult_proposer` learned router runs advisory | `:1473` |
| 3 | `_fuzzy_writeback` — verified facts pushed back to binding memory + SOM | `:1444` |
| 4 | on symbolic **miss**: `_fuzzy_propose_verify` — recall proposes, refuter disposes | `:1522` |
| wrap | cross-turn metacognition audit; contradiction/circular/unsupported → reply withheld | `:567` |

Before commit `800b71a` these were three subsystems that never met at runtime. Keep them joined.

`_fuzzy_pass` must call `train_lm_sequence_fused()`, **not** `perceive_text()` — the latter's
`predictor.step()` drops the attention gradient. Two entry points training different models
makes results irreproducible.

### STAMLAT

`crisp/engines/neural/stamlat_transformer.hpp`. **300,768 params** (d_model 80, 2 layers,
4 heads, ctx 64, vocab 408). Its role is **content-locked renderer**, not a language model:
crisp supplies the content, an allow-set masks the logits so it cannot emit a fact crisp
didn't hand it. Do not expand its role — at this size it template-matches and fills wrong
content (see `out/distill_unified3.log`).

## Trust the instruments, not the docs

`brain3/brain_development.log` is a 2255-line architecture journal. It is the best
module-by-module explanation of the system **and its scores are not trustworthy**:

- last entry is **2026-08-22**. Work after that exists only in git commit bodies, which are
  unusually detailed. `git log --format=%b` is the real changelog.
- its "192/192 100%" style claims were audited in commit `838880e` and found fabricated —
  the discovery engine assigned `r2_score`/`mse` as literal constants, so any dataset was
  reported as a verified law (measured false-discovery rate: 200/200).

**Project convention: verified, not asserted.** New capability ships with a held-out probe
in the same change.

| Instrument | What it tells you | Cost |
|---|---|---|
| `build_cmake/heldout_probe` | honest capability. Inputs seeded nowhere else. **73/92 = 79.3%** | ~5s |
| `build_cmake/gradcheck` | analytic gradients vs finite differences. 9/9 | fast |
| `ctest -E brain_eval` | **22/22 green** (2026-09-03). Note: `Testing/Temporary/LastTestsFailed.log` records the last run that *had* failures, not the last run — do not read current state from it | ~9 min |
| `brain_eval` | gate suite. **>10 min** — run it in a terminal, don't block a session on it | slow |
| `benchmarks/benchmark_evaluator.py` | 60/40 teach/query holdout (older versions scored rows right after teaching them) | net |

Use `heldout_probe` as the fast feedback signal.

## Build & run

```bash
cd brain3
cmake --build build_cmake --target heldout_probe   # or brain_master, gradcheck, …
ctest --test-dir build_cmake
./build_cmake/brain_master --interactive           # also --query --json-stream --ingest-all --sleep
```

Env flags: `BRAIN_NO_FUZZY=1` (revert to pre-integration behaviour, for A/B),
`BRAIN_NO_GAR_CACHE=1` (force graph-reasoner retrain instead of loading cached embeddings).

Python: **`/opt/homebrew/bin/python3.13`**. The repo `venv/` stalls under iCloud — don't use it.

## Known gaps (from heldout_probe)

- discovery: only multiplicative/power invariants. Affine `y=2x+1`, additive, exponential,
  polynomial all fail
- causal: L3 counterfactual returns the same value as L2 in some cases
- metacog: boundary invariants miss negative age/distance, p>1 → returns `VERIFIED_SOUND`
- `_fuzzy_propose_verify` proposes for symbols the fuzzy side has never seen

## Direction

Goal is standalone identity — brain3 is the intelligence, a small (~1B) LLM is at most a
bounded, swappable translator at the eyes/mouth boundary. It never decides truth, retrieves a
fact, or computes a value. Target domains: mathematics, research, coding. Not creativity,
not general chat.

Breadth (finance, vision, slang, Codeforces, philosophy curricula) is legacy exploration and
is not on the critical path.
