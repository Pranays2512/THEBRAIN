# Vault Architecture

This file contains the directory structure of the vault to help AI agents understand the repository structure without consuming excessive tokens.

## Folder Structure
```text
THEBRAIN/
├── .gitignore
├── .remember
│   ├── .gitignore
│   ├── logs
│   │   ├── autonomous
│   │   │   ├── save-000157.log
│   │   │   ├── save-000202.log
│   │   │   ├── save-000209.log
│   │   │   ├── save-000215.log
│   │   │   ├── save-000340.log
│   │   │   ├── save-000348.log
│   │   │   ├── save-000354.log
│   │   │   ├── save-000424.log
│   │   │   ├── save-000826.log
│   │   │   ├── save-001004.log
│   │   │   ├── save-001029.log
│   │   │   ├── save-001330.log
│   │   │   ├── save-001352.log
│   │   │   ├── save-001657.log
│   │   │   ├── save-001709.log
│   │   │   ├── save-001712.log
│   │   │   ├── save-001717.log
│   │   │   ├── save-001802.log
│   │   │   ├── save-001815.log
│   │   │   ├── save-001821.log
│   │   │   ├── save-001827.log
│   │   │   ├── save-001839.log
│   │   │   ├── save-001849.log
│   │   │   ├── save-001903.log
│   │   │   ├── save-001904.log
│   │   │   ├── save-001915.log
│   │   │   ├── save-001917.log
│   │   │   ├── save-001937.log
│   │   │   ├── save-001939.log
│   │   │   ├── save-001952.log
│   │   │   ├── save-002003.log
│   │   │   ├── save-002012.log
│   │   │   ├── save-002022.log
│   │   │   ├── save-002201.log
│   │   │   ├── save-002248.log
│   │   │   ├── save-002304.log
│   │   │   ├── save-002332.log
│   │   │   ├── save-002448.log
│   │   │   ├── save-002504.log
│   │   │   ├── save-002603.log
│   │   │   ├── save-002710.log
│   │   │   ├── save-003158.log
│   │   │   ├── save-003641.log
│   │   │   ├── save-003916.log
│   │   │   ├── save-005322.log
│   │   │   ├── save-010427.log
│   │   │   ├── save-010429.log
│   │   │   ├── save-010448.log
│   │   │   ├── save-010645.log
│   │   │   ├── save-010658.log
│   │   │   ├── save-010708.log
│   │   │   ├── save-011009.log
│   │   │   ├── save-011335.log
│   │   │   ├── save-011340.log
│   │   │   ├── save-011351.log
│   │   │   ├── save-011717.log
│   │   │   ├── save-011803.log
│   │   │   ├── save-012159.log
│   │   │   ├── save-012441.log
│   │   │   ├── save-012647.log
│   │   │   ├── save-012847.log
│   │   │   ├── save-013049.log
│   │   │   ├── save-013113.log
│   │   │   ├── save-013326.log
│   │   │   ├── save-013534.log
│   │   │   ├── save-013732.log
│   │   │   ├── save-013734.log
│   │   │   ├── save-013748.log
│   │   │   ├── save-013813.log
│   │   │   ├── save-013821.log
│   │   │   ├── save-013827.log
│   │   │   ├── save-014333.log
│   │   │   ├── save-014533.log
│   │   │   ├── save-014748.log
│   │   │   ├── save-015025.log
│   │   │   ├── save-015324.log
│   │   │   ├── save-015620.log
│   │   │   ├── save-015947.log
│   │   │   ├── save-020343.log
│   │   │   ├── save-020618.log
│   │   │   ├── save-020832.log
│   │   │   ├── save-021036.log
│   │   │   ├── save-021343.log
│   │   │   ├── save-021718.log
│   │   │   ├── save-022034.log
│   │   │   ├── save-022310.log
│   │   │   ├── save-022657.log
│   │   │   ├── save-022901.log
│   │   │   ├── save-090638.log
│   │   │   ├── save-091026.log
│   │   │   ├── save-091535.log
│   │   │   ├── save-144949.log
│   │   │   ├── save-145221.log
│   │   │   ├── save-145614.log
│   │   │   ├── save-150400.log
│   │   │   ├── save-150619.log
│   │   │   ├── save-150635.log
│   │   │   ├── save-150927.log
│   │   │   ├── save-151149.log
│   │   │   ├── save-151427.log
│   │   │   ├── save-151512.log
│   │   │   ├── save-151748.log
│   │   │   ├── save-151806.log
│   │   │   ├── save-152059.log
│   │   │   ├── save-152307.log
│   │   │   ├── save-152332.log
│   │   │   ├── save-152511.log
│   │   │   ├── save-152709.log
│   │   │   ├── save-152947.log
│   │   │   ├── save-153216.log
│   │   │   ├── save-153424.log
│   │   │   ├── save-153823.log
│   │   │   ├── save-154057.log
│   │   │   ├── save-154336.log
│   │   │   ├── save-154343.log
│   │   │   ├── save-154417.log
│   │   │   ├── save-154439.log
│   │   │   ├── save-154506.log
│   │   │   ├── save-154513.log
│   │   │   ├── save-154517.log
│   │   │   ├── save-154534.log
│   │   │   ├── save-154537.log
│   │   │   ├── save-154546.log
│   │   │   ├── save-154709.log
│   │   │   ├── save-154820.log
│   │   │   ├── save-155042.log
│   │   │   ├── save-155105.log
│   │   │   ├── save-155340.log
│   │   │   ├── save-155833.log
│   │   │   ├── save-160134.log
│   │   │   ├── save-160137.log
│   │   │   ├── save-160521.log
│   │   │   ├── save-161354.log
│   │   │   ├── save-162043.log
│   │   │   ├── save-162106.log
│   │   │   ├── save-162528.log
│   │   │   ├── save-162533.log
│   │   │   ├── save-162755.log
│   │   │   ├── save-162918.log
│   │   │   ├── save-163005.log
│   │   │   ├── save-163148.log
│   │   │   ├── save-163251.log
│   │   │   ├── save-163458.log
│   │   │   ├── save-163713.log
│   │   │   ├── save-163933.log
│   │   │   ├── save-164656.log
│   │   │   ├── save-165020.log
│   │   │   ├── save-165314.log
│   │   │   ├── save-165533.log
│   │   │   ├── save-165808.log
│   │   │   ├── save-170002.log
│   │   │   ├── save-170029.log
│   │   │   ├── save-170308.log
│   │   │   ├── save-170340.log
│   │   │   ├── save-170604.log
│   │   │   ├── save-170838.log
│   │   │   ├── save-171046.log
│   │   │   ├── save-171130.log
│   │   │   ├── save-172020.log
│   │   │   ├── save-172234.log
│   │   │   ├── save-172320.log
│   │   │   ├── save-172612.log
│   │   │   ├── save-172818.log
│   │   │   ├── save-172913.log
│   │   │   ├── save-173020.log
│   │   │   ├── save-173252.log
│   │   │   ├── save-173500.log
│   │   │   ├── save-173700.log
│   │   │   ├── save-173929.log
│   │   │   ├── save-174218.log
│   │   │   ├── save-174446.log
│   │   │   ├── save-174623.log
│   │   │   ├── save-174823.log
│   │   │   ├── save-174854.log
│   │   │   ├── save-175112.log
│   │   │   ├── save-175204.log
│   │   │   ├── save-175500.log
│   │   │   ├── save-175708.log
│   │   │   ├── save-175743.log
│   │   │   ├── save-175955.log
│   │   │   ├── save-175956.log
│   │   │   ├── save-180013.log
│   │   │   ├── save-180217.log
│   │   │   ├── save-180536.log
│   │   │   ├── save-184642.log
│   │   │   ├── save-185638.log
│   │   │   ├── save-190101.log
│   │   │   ├── save-190325.log
│   │   │   ├── save-190640.log
│   │   │   ├── save-190856.log
│   │   │   ├── save-191002.log
│   │   │   ├── save-191231.log
│   │   │   ├── save-191240.log
│   │   │   ├── save-191449.log
│   │   │   ├── save-192102.log
│   │   │   ├── save-192207.log
│   │   │   ├── save-192405.log
│   │   │   ├── save-192433.log
│   │   │   ├── save-192607.log
│   │   │   ├── save-193140.log
│   │   │   ├── save-194834.log
│   │   │   ├── save-195354.log
│   │   │   ├── save-200345.log
│   │   │   ├── save-204219.log
│   │   │   ├── save-204458.log
│   │   │   ├── save-204647.log
│   │   │   ├── save-205128.log
│   │   │   ├── save-205338.log
│   │   │   ├── save-205444.log
│   │   │   ├── save-205748.log
│   │   │   ├── save-210040.log
│   │   │   ├── save-210345.log
│   │   │   ├── save-210610.log
│   │   │   ├── save-210710.log
│   │   │   ├── save-211107.log
│   │   │   ├── save-211309.log
│   │   │   ├── save-211556.log
│   │   │   ├── save-211817.log
│   │   │   ├── save-211912.log
│   │   │   ├── save-212141.log
│   │   │   ├── save-212304.log
│   │   │   ├── save-212922.log
│   │   │   ├── save-212928.log
│   │   │   ├── save-212936.log
│   │   │   ├── save-213227.log
│   │   │   ├── save-214901.log
│   │   │   ├── save-215919.log
│   │   │   ├── save-220207.log
│   │   │   ├── save-231808.log
│   │   │   ├── save-232011.log
│   │   │   ├── save-232445.log
│   │   │   ├── save-233241.log
│   │   │   ├── save-233451.log
│   │   │   ├── save-234129.log
│   │   │   ├── save-234427.log
│   │   │   ├── save-234437.log
│   │   │   ├── save-234441.log
│   │   │   ├── save-234514.log
│   │   │   ├── save-234708.log
│   │   │   ├── save-234857.log
│   │   │   ├── save-234917.log
│   │   │   ├── save-234929.log
│   │   │   ├── save-235302.log
│   │   │   └── save-235917.log
│   │   ├── hook-errors.log
│   │   ├── logs-2026-05.tar.gz
│   │   ├── logs-2026-06.tar.gz
│   │   ├── memory-2026-06-27.log
│   │   ├── memory-2026-06-28.log
│   │   ├── memory-2026-06-29.log
│   │   ├── memory-2026-06-30.log
│   │   ├── memory-2026-07-02.log
│   │   ├── memory-2026-07-03.log
│   │   ├── memory-2026-07-04.log
│   │   └── memory-2026-07-05.log
│   ├── now.md
│   ├── remember.md
│   ├── tmp
│   │   ├── last-ndc.ts
│   │   ├── last-save-ts
│   │   ├── last-save.json
│   │   └── save-session.pid
│   ├── today-2026-05-03.done.md
│   └── today-2026-07-03.md
├── Brain
│   ├── .obsidian
│   │   ├── app.json
│   │   ├── appearance.json
│   │   ├── core-plugins.json
│   │   ├── graph.json
│   │   └── workspace.json
│   ├── Untitled.canvas
│   └── Welcome.md
├── README.md
├── brain2
│   ├── .cache
│   │   └── clangd
│   │       └── index
│   │           ├── analogy.hpp.5FFB6ACE2CFAA0A8.idx
│   │           ├── attention.hpp.FBBB06998F401C86.idx
│   │           ├── basal_ganglia.hpp.EFD349328973C5F3.idx
│   │           ├── binding_memory.hpp.654CDC6F62640244.idx
│   │           ├── brain.hpp.8908B7E11F25A690.idx
│   │           ├── brain2.cpp.DCF37C88C1773C87.idx
│   │           ├── cuda_math.cuh.8466E56C9AD7D710.idx
│   │           ├── decoder.hpp.E9E8AA50257DB939.idx
│   │           ├── emotion.hpp.0AAB4B76F49A474A.idx
│   │           ├── episodic.hpp.33BD85A452633FD6.idx
│   │           ├── global_workspace.hpp.8D868C657A8A787D.idx
│   │           ├── hierarchical_predictor.hpp.F636A7CEB932AA36.idx
│   │           ├── imagination.hpp.60D1D63DAD1240DE.idx
│   │           ├── language.hpp.B4AF606C48AD80CD.idx
│   │           ├── predictive_coding.hpp.116EBEC86395CA5A.idx
│   │           ├── predictor.hpp.3E9A94012F2AFE34.idx
│   │           ├── procedural_memory.hpp.A93801D45DBFC6FA.idx
│   │           ├── reasoning.hpp.87A45DE0DEF859FD.idx
│   │           ├── scratchpad.hpp.834388EE28067F4C.idx
│   │           ├── self_model.hpp.31E8C0FA87A01F36.idx
│   │           ├── som.hpp.E5424CC2ADCF36AB.idx
│   │           ├── symbolic.hpp.5019D52105FACFE2.idx
│   │           └── working_mem.hpp.299C373B374A6630.idx
│   ├── .gitignore
│   ├── 100_questions.txt
│   ├── CMakeLists.txt
│   ├── CONNECTIVITY.txt
│   ├── MILESTONES.md
│   ├── README.md
│   ├── agent.py
│   ├── algebra_engine.py
│   ├── analogy_engine.py
│   ├── analogy_struct.py
│   ├── appraisal_engine.py
│   ├── architecture_flaws.md
│   ├── architecture_roadmap.md
│   ├── autonomous_loop.py
│   ├── brain2
│   ├── brain2.cpp
│   ├── brain2.cpython-313-darwin.so
│   ├── brain2_architecture.md
│   ├── brain_chat.py
│   ├── brain_codegen.py
│   ├── brain_data.py
│   ├── brain_planner.py
│   ├── brain_repl.py
│   ├── brain_session.py
│   ├── brain_store
│   │   ├── facts.json
│   │   ├── functions.json
│   │   └── policies.json
│   ├── brain_store.py
│   ├── calculus_engine.py
│   ├── chat.py
│   ├── check_library
│   │   └── invariants.json
│   ├── check_library.py
│   ├── checkpoints
│   │   ├── executive_brain
│   │   │   ├── bg.bin
│   │   │   ├── binding.bin
│   │   │   ├── decoder.bin
│   │   │   ├── emotion.bin
│   │   │   ├── episodic.bin
│   │   │   ├── hpred.bin
│   │   │   ├── language.bin
│   │   │   ├── predictor.bin
│   │   │   ├── procedures.bin
│   │   │   ├── self.bin
│   │   │   ├── som.bin
│   │   │   ├── som.bin.tlb
│   │   │   ├── symbolic.bin
│   │   │   └── training_state.json
│   │   └── squad_run
│   │       ├── bg.bin
│   │       ├── binding.bin
│   │       ├── decoder.bin
│   │       ├── emotion.bin
│   │       ├── episodic.bin
│   │       ├── hpred.bin
│   │       ├── language.bin
│   │       ├── predictor.bin
│   │       ├── procedures.bin
│   │       ├── self.bin
│   │       ├── som.bin
│   │       ├── som.bin.tlb
│   │       └── symbolic.bin
│   ├── clear_ep.cpp
│   ├── code_gen.py
│   ├── component_validation.py
│   ├── composable_proposer.py
│   ├── composable_synth.py
│   ├── compositional.py
│   ├── concept_blend.py
│   ├── concept_memory.py
│   ├── conceptnet_taxonomy.py
│   ├── conjecture_sandbox.py
│   ├── context_embed.py
│   ├── conversation_engine.py
│   ├── core
│   │   ├── analogy.hpp
│   │   ├── attention.hpp
│   │   ├── basal_ganglia.hpp
│   │   ├── binding_memory.hpp
│   │   ├── brain.hpp
│   │   ├── cuda_math.cu
│   │   ├── cuda_math.cuh
│   │   ├── debug.hpp
│   │   ├── decoder.hpp
│   │   ├── emotion.hpp
│   │   ├── episodic.hpp
│   │   ├── factorizer.hpp
│   │   ├── global_workspace.hpp
│   │   ├── hierarchical_predictor.hpp
│   │   ├── imagination.hpp
│   │   ├── invariants.hpp
│   │   ├── language.hpp
│   │   ├── logic_engine.hpp
│   │   ├── lsh.hpp
│   │   ├── memoization.hpp
│   │   ├── policy_engine.hpp
│   │   ├── predictive_coding.hpp
│   │   ├── predictor.hpp
│   │   ├── procedural_memory.hpp
│   │   ├── proposer.hpp
│   │   ├── reasoning.hpp
│   │   ├── reasoning_ops.hpp
│   │   ├── refuter.hpp
│   │   ├── regularity.hpp
│   │   ├── scratchpad.hpp
│   │   ├── self_model.hpp
│   │   ├── som.hpp
│   │   ├── sparse_lstm.hpp
│   │   ├── sparse_tensor.hpp
│   │   ├── symbolic.hpp
│   │   └── working_mem.hpp
│   ├── core_knowledge.py
│   ├── corpus_scale.py
│   ├── coverage_harness.py
│   ├── cpp_accel.py
│   ├── crispify_bridge.py
│   ├── curiosity_cross.py
│   ├── curiosity_loop.py
│   ├── data
│   │   ├── alice.txt
│   │   ├── bridge_corpus.json
│   │   ├── conversational_corpus.json
│   │   ├── conversational_massive.json
│   │   ├── english1.txt
│   │   ├── english2.txt
│   │   ├── english3.txt
│   │   ├── english4.txt
│   │   ├── english5.txt
│   │   ├── english6.txt
│   │   ├── english7.txt
│   │   ├── english8.txt
│   │   ├── generate_bridge.py
│   │   ├── generate_conversational.py
│   │   ├── generate_math.py
│   │   ├── kimi_data.txt
│   │   ├── math1.txt
│   │   ├── math2.txt
│   │   ├── math3.txt
│   │   ├── math4.txt
│   │   ├── math5.txt
│   │   ├── math6.txt
│   │   ├── math7.txt
│   │   ├── math8.txt
│   │   ├── math9.txt
│   │   ├── math_corpus.json
│   │   ├── new_data.txt
│   │   ├── science3.txt
│   │   ├── science4.txt
│   │   ├── science5.txt
│   │   ├── science6.txt
│   │   ├── science7.txt
│   │   ├── science8.txt
│   │   ├── simple_stories.txt
│   │   ├── squad_qa.json
│   │   ├── squad_test_6k.json
│   │   ├── squad_train_80k.json
│   │   ├── ssc6.txt
│   │   ├── ssc7.txt
│   │   ├── ssc8.txt
│   │   ├── test_sentences_6k.txt
│   │   ├── text8
│   │   ├── text8.zip
│   │   ├── train-v2.0.json
│   │   └── train_sentences_80k.txt
│   ├── debug_brain.cpp
│   ├── debug_wm.cpp
│   ├── deeper_grammar.py
│   ├── dimensional_verify.py
│   ├── discourse.py
│   ├── distill_data.jsonl
│   ├── docs
│   │   ├── kimi_data_prompt.txt
│   │   ├── kimi_data_prompt_v2.txt
│   │   └── ncert_extract_prompt.txt
│   ├── domain_features.py
│   ├── dp_greedy_synth.py
│   ├── dp_proposer.py
│   ├── dual_process.py
│   ├── dual_process_engine.py
│   ├── event_form.py
│   ├── event_parse.py
│   ├── event_predict.py
│   ├── event_verify.py
│   ├── exam.py
│   ├── exam_math.py
│   ├── fact_extractor.py
│   ├── factorizer.py
│   ├── feature_learner.py
│   ├── generate_data.py
│   ├── generate_massive_data.py
│   ├── generate_more_data.py
│   ├── glove.6B.100d.txt
│   ├── glove.6B.50d.txt
│   ├── glove.6B.zip
│   ├── ground_blend.py
│   ├── ground_numeric.py
│   ├── ground_reason.py
│   ├── ground_to_binding.py
│   ├── grounding.py
│   ├── harden_regress.py
│   ├── harden_test.py
│   ├── inductive_engine.py
│   ├── integral_engine.py
│   ├── invariant_miner.py
│   ├── irregularity_detector.py
│   ├── knowledge_base.py
│   ├── knowledge_distill.py
│   ├── knowledge_engine.py
│   ├── knowledge_pack.py
│   ├── learn_by_reading.py
│   ├── learned_guidance.py
│   ├── llm_adapter.py
│   ├── llm_extractor.py
│   ├── loop_synth.py
│   ├── loop_synth2.py
│   ├── loop_synth3.py
│   ├── loop_synth4.py
│   ├── math_chat.py
│   ├── math_parser.py
│   ├── math_synth.py
│   ├── means_ends.py
│   ├── mouth.py
│   ├── nested_parser.py
│   ├── neural_lm.py
│   ├── neural_lm_torch.py
│   ├── neuro_bridge.py
│   ├── nl_front.py
│   ├── online_proposer.py
│   ├── online_proposer2.py
│   ├── ontology_dataset.txt
│   ├── open_world.py
│   ├── parse_template.py
│   ├── physics_engine.py
│   ├── planning_engine.py
│   ├── policy_induction.py
│   ├── policy_proposer.py
│   ├── prob_compute.py
│   ├── program_synth.py
│   ├── program_synth_guided.py
│   ├── program_synth_policy.py
│   ├── program_synth_tree.py
│   ├── query_planner.py
│   ├── reading_loop.py
│   ├── reasoning_engine.py
│   ├── reasoning_suite.py
│   ├── refute_synth.py
│   ├── refuter.py
│   ├── scorecard.json
│   ├── scorecard_baseline.json
│   ├── scripts
│   │   └── train_text_stream.py
│   ├── semantic_depth.py
│   ├── semantic_memory.py
│   ├── server.py
│   ├── simulate_op.txt
│   ├── squad_run_log.json
│   ├── static
│   │   ├── app.js
│   │   ├── index.html
│   │   └── style.css
│   ├── stress_synth.py
│   ├── structural_parser.py
│   ├── student_trainer.py
│   ├── synth_engine.py
│   ├── synth_invariant.py
│   ├── synthesis_engine.py
│   ├── template_memory.py
│   ├── test_100.txt
│   ├── test_700.txt
│   ├── test_analogy.cpp
│   ├── test_brain.cpp
│   ├── test_causal.sh
│   ├── test_convo.txt
│   ├── test_math_logic.cpp
│   ├── test_open_lang.py
│   ├── test_phase_a.py
│   ├── test_rint.cpp
│   ├── test_snprintf.cpp
│   ├── tests
│   │   ├── __init__.py
│   │   ├── benchmark_speed.py
│   │   ├── generate_hardened_suite.py
│   │   ├── run_all.py
│   │   ├── run_hardened_suite.py
│   │   ├── test_algebra_engine.py
│   │   ├── test_analogy.py
│   │   ├── test_analogy_engine.py
│   │   ├── test_appraisal_engine.py
│   │   ├── test_arch_fixes.py
│   │   ├── test_attention.py
│   │   ├── test_bg_planning.py
│   │   ├── test_brain_chat.py
│   │   ├── test_brain_session.py
│   │   ├── test_calculus_engine.py
│   │   ├── test_causal_reasoning.py
│   │   ├── test_code_gen.py
│   │   ├── test_confidence.py
│   │   ├── test_consolidation.py
│   │   ├── test_conversation_engine.py
│   │   ├── test_core_knowledge.py
│   │   ├── test_curiosity_loop.py
│   │   ├── test_discovery.py
│   │   ├── test_dual_process.py
│   │   ├── test_emotion.py
│   │   ├── test_episodic.py
│   │   ├── test_eval_harness.py
│   │   ├── test_fact_extractor.py
│   │   ├── test_hardened_1100.txt
│   │   ├── test_imagination.py
│   │   ├── test_inductive_engine.py
│   │   ├── test_integral_engine.py
│   │   ├── test_integration.py
│   │   ├── test_knowledge_base.py
│   │   ├── test_knowledge_engine.py
│   │   ├── test_knowledge_pack.py
│   │   ├── test_language.py
│   │   ├── test_learned_guidance.py
│   │   ├── test_llm_adapter.py
│   │   ├── test_llm_extractor.py
│   │   ├── test_math_chat.py
│   │   ├── test_math_parser.py
│   │   ├── test_multihop.py
│   │   ├── test_neuro_bridge.py
│   │   ├── test_new_features.py
│   │   ├── test_physics_engine.py
│   │   ├── test_planning_engine.py
│   │   ├── test_planning_tree.py
│   │   ├── test_predictive_logic.py
│   │   ├── test_predictor.py
│   │   ├── test_query_planner.py
│   │   ├── test_reasoning_engine.py
│   │   ├── test_scratchpad.py
│   │   ├── test_search_engine.py
│   │   ├── test_self_model.py
│   │   ├── test_semantic_memory.py
│   │   ├── test_som.py
│   │   ├── test_symbolic.py
│   │   ├── test_synthesis_engine.py
│   │   ├── test_td_learning.py
│   │   └── test_working_mem.py
│   ├── tiny_shakespeare.txt
│   ├── train
│   │   ├── concept_encoder.py
│   │   ├── conceptnet-assertions-5.7.0.csv.gz
│   │   ├── conceptnet_en_subset.json
│   │   ├── conceptnet_loader.py
│   │   ├── eval_v3.py
│   │   ├── math_sequences.py
│   │   ├── preflight.py
│   │   └── train.py
│   ├── train_from_data.py
│   ├── train_pipeline.py
│   ├── trained
│   │   ├── owned_lm.pt
│   │   ├── owned_lm_data.pt
│   │   └── teacher_cache.json
│   ├── tree_domains.py
│   ├── tree_learn.py
│   ├── tree_reason.py
│   ├── type_oracle.py
│   ├── validate.py
│   ├── verb_learn.py
│   ├── verifier_monitor.py
│   ├── whole_brain.py
│   ├── word_math.py
│   └── world_knowledge.py
├── generate_architecture_map.py
├── ncert_class1_english
│   └── aeen1dd.zip
└── vault_architecture.md
```
