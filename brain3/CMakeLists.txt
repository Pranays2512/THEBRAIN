cmake_minimum_required(VERSION 3.16)
project(Brain3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Include Directories ──────────────────────────────────────────────────────
# All headers are included relative to the brain3 root. No more -I flags needed.
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/fuzzy/core
    ${CMAKE_SOURCE_DIR}/crisp/core
    ${CMAKE_SOURCE_DIR}/fuzzy
    ${CMAKE_SOURCE_DIR}/crisp
)

# ── Platform: macOS Accelerate Framework ─────────────────────────────────────
if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    set(PLATFORM_LIBS ${ACCELERATE_FRAMEWORK})
    add_compile_options(-Wno-deprecated-declarations)
else()
    set(PLATFORM_LIBS "")
endif()

# ── JNI Bridge ───────────────────────────────────────────────────────────────
# On macOS, find_package(JNI) requires AWT which is not needed.
# Use JAVA_HOME directly to find JNI headers.
if(DEFINED ENV{JAVA_HOME})
    set(_JDK "$ENV{JAVA_HOME}")
elseif(APPLE)
    execute_process(COMMAND /usr/libexec/java_home
                    OUTPUT_VARIABLE _JDK OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

if(_JDK AND EXISTS "${_JDK}/include/jni.h")
    set(JNI_FOUND TRUE)
    set(JNI_INCLUDE_DIRS "${_JDK}/include" "${_JDK}/include/darwin" "${_JDK}/include/linux")
    include_directories(${JNI_INCLUDE_DIRS})
    add_library(brainjni SHARED
        adapters/jni/BrainJNI.cpp
    )
    target_link_libraries(brainjni ${PLATFORM_LIBS})
    set_target_properties(brainjni PROPERTIES OUTPUT_NAME "brainjni")
    message(STATUS "JNI bridge enabled: ${_JDK}")
else()
    message(WARNING "JNI not found — skipping BrainJNI.cpp bridge build")
endif()

# ── Helper: add a phase test ─────────────────────────────────────────────────
function(add_phase_test name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} ${PLATFORM_LIBS})
endfunction()

# ── Test Targets ─────────────────────────────────────────────────────────────
add_phase_test(test_phase1_synth      test_phase1_synth.cpp)
add_phase_test(test_phase2_synth      test_phase2_synth.cpp)
add_phase_test(test_phase3_synth      test_phase3_synth.cpp)
add_phase_test(test_phase4_synth      test_phase4_synth.cpp)
add_phase_test(test_phase5_synth      test_phase5_synth.cpp)
add_phase_test(test_phase6_synth      test_phase6_synth.cpp)
add_phase_test(test_phase7_reasoning  test_phase7_reasoning.cpp)
add_phase_test(test_phase8_integration test_phase8_integration.cpp)

# ── STAMLAT v2 (trainable micro-LM) ───────────────────────────────────────────
add_phase_test(test_stamlat_transformer test_stamlat_transformer.cpp)
add_executable(demo_stamlat_chat demo_stamlat_chat.cpp)
target_include_directories(demo_stamlat_chat PRIVATE .)
target_link_libraries(demo_stamlat_chat PRIVATE ${PLATFORM_LIBS})

# ── STAMLAT mouth v3 (word tokens + KV-cache streaming + constrained sampling) ─
add_phase_test(test_stamlat_streaming test_stamlat_streaming.cpp)
add_executable(demo_stamlat_mouth demo_stamlat_mouth.cpp)
target_include_directories(demo_stamlat_mouth PRIVATE .)
target_link_libraries(demo_stamlat_mouth PRIVATE ${PLATFORM_LIBS})

# ── STAMLAT voice (emotion coupling + style loop / Discovery-for-voice) ───────
add_phase_test(test_stamlat_voice test_stamlat_voice.cpp)
add_executable(demo_stamlat_voice demo_stamlat_voice.cpp)
target_include_directories(demo_stamlat_voice PRIVATE .)
target_link_libraries(demo_stamlat_voice PRIVATE ${PLATFORM_LIBS})
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(test_stamlat_voice PRIVATE -O2)
    target_compile_options(demo_stamlat_voice PRIVATE -O2)
endif()

# ── Plan-conditioned mouth (Sprint 1 falsifiers) ──────────────────────────────
add_phase_test(test_plan_mouth test_plan_mouth.cpp)
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(test_plan_mouth PRIVATE -O2)
endif()

# ── Native mouth (production mount) ───────────────────────────────────────────
add_phase_test(test_native_mouth test_native_mouth.cpp)
add_executable(brain_eval eval/run_eval.cpp)
target_include_directories(brain_eval PRIVATE .)
target_link_libraries(brain_eval PRIVATE ${PLATFORM_LIBS})
target_compile_options(brain_eval PRIVATE -O2)
add_executable(train_native_mouth train_native_mouth.cpp)
target_include_directories(train_native_mouth PRIVATE .)
target_link_libraries(train_native_mouth PRIVATE ${PLATFORM_LIBS})
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(test_native_mouth PRIVATE -O2)
    target_compile_options(train_native_mouth PRIVATE -O2)
endif()

# ── Mouth ctx-scaling benchmark ───────────────────────────────────────────────
add_executable(bench_stamlat_ctx benchmarks/bench_stamlat_ctx.cpp)
target_include_directories(bench_stamlat_ctx PRIVATE .)
target_link_libraries(bench_stamlat_ctx PRIVATE ${PLATFORM_LIBS})
target_compile_options(bench_stamlat_ctx PRIVATE -O2)

# ── Exactness scoreboard (Sprint 3) ───────────────────────────────────────────
add_executable(exactness_scoreboard eval/exactness_scoreboard.cpp)
target_include_directories(exactness_scoreboard PRIVATE .)
target_link_libraries(exactness_scoreboard PRIVATE ${PLATFORM_LIBS})
target_compile_options(exactness_scoreboard PRIVATE -O2)

# ── Learned intent router (Sprint 4a) ─────────────────────────────────────────
add_phase_test(test_intent_router test_intent_router.cpp)
target_compile_options(test_intent_router PRIVATE -O2)

# ── Curiosity diet + contradiction quarantine (Sprint 5b) ─────────────────────
add_phase_test(test_curiosity_diet test_curiosity_diet.cpp)
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(test_curiosity_diet PRIVATE -O2)
endif()

# ── Metacognition + sentiment perception ──────────────────────────────────────
add_phase_test(test_metacognition test_metacognition.cpp)

# ── Held-out capability probe (measures generalization, not recall) ──────────
add_executable(heldout_probe eval/heldout_probe.cpp)
target_include_directories(heldout_probe PRIVATE .)
target_link_libraries(heldout_probe PRIVATE ${PLATFORM_LIBS})
target_compile_options(heldout_probe PRIVATE -O2)

# ── Readiness assessment (the honest scorecard) ──────────────────────────────
add_executable(eval_readiness eval/readiness.cpp)
target_include_directories(eval_readiness PRIVATE .)
target_link_libraries(eval_readiness PRIVATE ${PLATFORM_LIBS})
target_compile_options(eval_readiness PRIVATE -O2)

# ── Intuition fixes falsifiers ────────────────────────────────────────────────
add_phase_test(test_intuition_fixed test_intuition_fixed.cpp)
add_executable(swarm_train tools/swarm_train.cpp)
target_include_directories(swarm_train PRIVATE .)
target_link_libraries(swarm_train PRIVATE ${PLATFORM_LIBS})
target_compile_options(swarm_train PRIVATE -O2)
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(test_intuition_fixed PRIVATE -O2)
endif()

# ── Fuzzy ingestion pipeline (Sprint 5) ───────────────────────────────────────
add_phase_test(test_fuzzy_ingestion test_fuzzy_ingestion.cpp)
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(test_fuzzy_ingestion PRIVATE -O2)
endif()

# ── Sleep kernel (Sprint 4c) ──────────────────────────────────────────────────
add_phase_test(test_sleep_kernel test_sleep_kernel.cpp)
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(test_sleep_kernel PRIVATE -O2)
endif()

# ── Graph-attention reasoner (Sprint 4b) ──────────────────────────────────────
add_phase_test(test_graph_reasoner test_graph_reasoner.cpp)
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(test_graph_reasoner PRIVATE -O2)
endif()
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(test_stamlat_streaming PRIVATE -O2)
    target_compile_options(demo_stamlat_mouth PRIVATE -O2)
endif()
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(test_stamlat_transformer PRIVATE -O2)
    target_compile_options(demo_stamlat_chat PRIVATE -O2)
endif()

# ── Per-engine test targets ───────────────────────────────────────────────────
add_phase_test(test_math_engines
    crisp/engines/math/math_full_test.cpp)

add_phase_test(code_engine_test
    crisp/engines/code/code_engine_test.cpp)

add_phase_test(crisp_reasoning_test
    crisp/engines/reasoning/crisp_reasoning_test.cpp)

add_phase_test(crisp_store_test
    crisp/engines/store/crisp_store_test.cpp)

add_phase_test(crisp_knowledge_test
    crisp/engines/knowledge/crisp_knowledge_test.cpp)

add_phase_test(crisp_math_test
    crisp/engines/math/crisp_math_test.cpp)

# ── Demos ─────────────────────────────────────────────────────────────────────
add_executable(demo_vision demo_vision.cpp)
target_include_directories(demo_vision PRIVATE .)
target_link_libraries(demo_vision PRIVATE ${PLATFORM_LIBS})

add_executable(demo_general_play demo_general_play.cpp)
target_include_directories(demo_general_play PRIVATE .)
target_link_libraries(demo_general_play PRIVATE ${PLATFORM_LIBS})

add_executable(demo_self_play demo_self_play.cpp)
target_include_directories(demo_self_play PRIVATE .)
target_link_libraries(demo_self_play PRIVATE ${PLATFORM_LIBS})

add_executable(demo_sleep demo_sleep.cpp)
target_include_directories(demo_sleep PRIVATE .)
target_link_libraries(demo_sleep PRIVATE ${PLATFORM_LIBS})

add_executable(demo_intuition demo_intuition.cpp)
target_include_directories(demo_intuition PRIVATE .)
target_link_libraries(demo_intuition PRIVATE ${PLATFORM_LIBS})

add_executable(demo_neuro_symbolic demo_neuro_symbolic.cpp)
target_include_directories(demo_neuro_symbolic PRIVATE .)
target_link_libraries(demo_neuro_symbolic PRIVATE ${PLATFORM_LIBS})

add_executable(demo_cognitive_bridge demo_cognitive_bridge.cpp)
target_include_directories(demo_cognitive_bridge PRIVATE .)
target_link_libraries(demo_cognitive_bridge PRIVATE ${PLATFORM_LIBS})

add_executable(demo_batch_training demo_batch_training.cpp)
target_include_directories(demo_batch_training PRIVATE .)
target_link_libraries(demo_batch_training PRIVATE ${PLATFORM_LIBS})

add_executable(demo_language_training demo_language_training.cpp)
target_include_directories(demo_language_training PRIVATE .)
target_link_libraries(demo_language_training PRIVATE ${PLATFORM_LIBS})

add_executable(demo_brain2_dataset demo_brain2_dataset.cpp)
target_include_directories(demo_brain2_dataset PRIVATE .)
target_link_libraries(demo_brain2_dataset PRIVATE ${PLATFORM_LIBS})

add_executable(demo_curriculum_training demo_curriculum_training.cpp)
target_include_directories(demo_curriculum_training PRIVATE .)
target_link_libraries(demo_curriculum_training PRIVATE ${PLATFORM_LIBS})

add_executable(demo_glove_import demo_glove_import.cpp)
target_include_directories(demo_glove_import PRIVATE .)
target_link_libraries(demo_glove_import PRIVATE ${PLATFORM_LIBS})

add_executable(demo_native_chat demo_native_chat.cpp)
target_include_directories(demo_native_chat PRIVATE .)
target_link_libraries(demo_native_chat PRIVATE ${PLATFORM_LIBS})

add_executable(train_brain train_brain.cpp)
target_include_directories(train_brain PRIVATE .)
target_link_libraries(train_brain PRIVATE ${PLATFORM_LIBS})

add_executable(brain_master core/master_orchestrator.cpp)
target_include_directories(brain_master PRIVATE .)
target_link_libraries(brain_master PRIVATE ${PLATFORM_LIBS})

# ── CTest integration ─────────────────────────────────────────────────────────
enable_testing()
add_test(NAME phase1  COMMAND test_phase1_synth)
add_test(NAME phase2  COMMAND test_phase2_synth)
add_test(NAME phase3  COMMAND test_phase3_synth)
add_test(NAME phase4  COMMAND test_phase4_synth)
add_test(NAME phase5  COMMAND test_phase5_synth)
add_test(NAME phase6  COMMAND test_phase6_synth)
add_test(NAME phase7  COMMAND test_phase7_reasoning)
add_test(NAME phase8  COMMAND test_phase8_integration)
add_test(NAME code_engine  COMMAND code_engine_test)
add_test(NAME crisp_store  COMMAND crisp_store_test)
add_test(NAME stamlat_transformer COMMAND test_stamlat_transformer)
add_test(NAME stamlat_streaming  COMMAND test_stamlat_streaming)
add_test(NAME stamlat_voice      COMMAND test_stamlat_voice)
add_test(NAME native_mouth       COMMAND test_native_mouth)
add_test(NAME plan_mouth         COMMAND test_plan_mouth)
add_test(NAME brain_eval         COMMAND brain_eval)
add_test(NAME intent_router      COMMAND test_intent_router)
add_test(NAME fuzzy_ingestion    COMMAND test_fuzzy_ingestion)
add_test(NAME curiosity_diet     COMMAND test_curiosity_diet)
add_test(NAME graph_reasoner     COMMAND test_graph_reasoner)
add_test(NAME sleep_kernel       COMMAND test_sleep_kernel)

# ── Summary ───────────────────────────────────────────────────────────────────
message(STATUS "Brain3 configured.")
message(STATUS "  Include root: ${CMAKE_SOURCE_DIR}")
message(STATUS "  Platform libs: ${PLATFORM_LIBS}")
message(STATUS "  Build dir: ${CMAKE_BINARY_DIR}")
message(STATUS "  Run: cmake --build build && ctest --test-dir build")
