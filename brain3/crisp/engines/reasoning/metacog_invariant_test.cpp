/**
 * metacog_invariant_test.cpp — physical & mathematical boundary invariants.
 *
 * The metacognitive refuter is the load-bearing half of "fuzzy proposes, crisp
 * disposes": it is what stops an associative recall from being presented as a
 * fact. It knew that mass must be positive and that a divisor must be nonzero,
 * and nothing else — so `age = -5`, `distance = -12` and `probability = 1.7`
 * all came back VERIFIED_SOUND (heldout_probe section G).
 *
 * The negative cases below matter more than the positive ones. A refuter that
 * rejects everything is as useless as one that accepts everything, and an
 * over-eager bound is worse than a missing one because it silently blocks
 * correct facts. `temperature = -50` is the deliberate example: perfectly valid
 * in Celsius, invalid in Kelvin, and the engine does not know the unit — so it
 * must NOT guess.
 */
#include <cstdio>
#include <string>

#include "crisp/engines/reasoning/metacognitive_engine.hpp"

using namespace brain2::reasoning;

static int g_pass = 0, g_fail = 0;

static void ok(bool cond, const std::string& label, const std::string& note = "") {
    if (cond) { ++g_pass; std::printf("  PASS  %-46s %s\n", label.c_str(), note.c_str()); }
    else      { ++g_fail; std::printf("  FAIL  %-46s %s\n", label.c_str(), note.c_str()); }
}

static void refutes(const std::string& subj, const std::string& rel,
                    const std::string& obj, const std::string& label) {
    MetacognitiveEngine mce;
    auto v = mce.refute(subj, rel, obj, nullptr, nullptr);
    ok(v.is_refuted, label, v.is_refuted ? v.corrected_truth
                                         : "NOT refuted — verdict: " + v.verdict_str);
}

static void accepts(const std::string& subj, const std::string& rel,
                    const std::string& obj, const std::string& label) {
    MetacognitiveEngine mce;
    auto v = mce.refute(subj, rel, obj, nullptr, nullptr);
    ok(!v.is_refuted, label, v.is_refuted ? "WRONGLY refuted: " + v.falsification_reason
                                          : "sound");
}

int main() {
    std::printf("\n=== metacognition: boundary invariants ===\n\n");

    std::printf("1. Must refute — impossible values\n");
    refutes("age",         "val", "-5",   "negative age");
    refutes("distance",    "val", "-12",  "negative distance");
    refutes("probability", "val", "1.7",  "probability above 1");
    refutes("probability", "val", "-0.2", "probability below 0");
    refutes("length",      "val", "-3",   "negative length");
    refutes("speed",       "val", "-8",   "negative speed (magnitude, not velocity)");

    std::printf("\n2. Must still refute — the two that already worked\n");
    refutes("mass",    "val", "-3", "negative mass (regression)");
    refutes("divisor", "val", "0",  "division by zero (regression)");

    std::printf("\n3. Must NOT refute — legitimate values\n");
    accepts("age",         "val", "30",   "positive age");
    accepts("distance",    "val", "12",   "positive distance");
    accepts("probability", "val", "0.5",  "probability in range");
    accepts("probability", "val", "1",    "probability exactly 1 (inclusive bound)");
    accepts("probability", "val", "0",    "probability exactly 0 (inclusive bound)");
    accepts("age",         "val", "0",    "age zero is valid (newborn)");
    accepts("mass",        "val", "3",    "positive mass");
    accepts("divisor",     "val", "2",    "nonzero divisor");

    std::printf("\n4. Must NOT guess where units are ambiguous\n");
    accepts("temperature", "val", "-50",  "-50 degrees: valid Celsius, invalid Kelvin");
    accepts("altitude",    "val", "-30",  "below sea level is real");
    accepts("balance",     "val", "-500", "an account can be overdrawn");
    accepts("velocity",    "val", "-8",   "velocity is signed, unlike speed");

    std::printf("\n5. Non-numeric objects must pass through untouched\n");
    accepts("age",  "val", "unknown", "non-numeric object");
    accepts("mass", "isa", "quantity", "relation that is not a value assignment");

    std::printf("\n=== %d passed, %d failed ===\n\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
