/**
 * brain3/crisp/engines/math/unsolved_grand_challenges_evaluator.cpp
 *
 * Driver and execution harness for evaluating The Brain against all major unsolved grand challenges.
 */

#include "unsolved_grand_challenges_evaluator.hpp"
#include <iostream>
#include <iomanip>

using namespace thebrain::grand_challenges;

std::string provenance_to_string(EpistemicProvenance p) {
    switch (p) {
        case EpistemicProvenance::REPRODUCED_HISTORICAL_LITERATURE_RESULT:
            return "REPRODUCED_HISTORICAL_LITERATURE";
        case EpistemicProvenance::COMPUTATIONAL_INSTANCE_SOLVER:
            return "COMPUTATIONAL_INSTANCE_SOLVER";
        case EpistemicProvenance::SPECULATIVE_HEURISTIC_MODEL:
            return "SPECULATIVE_HEURISTIC_MODEL";
    }
}

void print_evaluation_report(const GrandChallengeEvaluation& eval, int index) {
    std::cout << "\n==========================================================================\n";
    std::cout << "🔬 " << index << ". " << eval.problem_name << "\n";
    std::cout << "==========================================================================\n";
    std::cout << "• Field of Science       : " << eval.field_of_science << "\n";
    std::cout << "• Classical Status       : " << eval.millennium_or_historical_status << "\n";
    std::cout << "• Literature Benchmarks  : " << eval.known_literature_benchmarks << "\n";
    std::cout << "• Evaluation Time        : " << std::fixed << std::setprecision(2) << eval.computation_time_ms << " ms\n\n";

    std::cout << "🧪 Test Cases Evaluated (" << eval.test_cases.size() << " cases):\n";
    for (const auto& tc : eval.test_cases) {
        std::cout << "   " << (tc.passed ? "✅ [PASSED]" : "❌ [FAILED]") << " " << tc.case_id << "\n";
        std::cout << "      Provenance : [" << provenance_to_string(tc.provenance) << "]\n";
        std::cout << "      Reference  : " << tc.literature_reference << "\n";
        std::cout << "      Result     : " << tc.exact_output << "\n";
    }

    std::cout << "\n📖 What The Brain Computes / Encodes:\n";
    std::cout << eval.what_the_brain_computes_or_verifies << "\n";

    std::cout << "\n⚠️ What Remains Open & Identified Bottleneck:\n";
    std::cout << "• Remaining Gap: " << eval.what_remains_open << "\n";
    std::cout << "• Bottleneck   : " << eval.exact_bottleneck_barrier << "\n";

    std::cout << "\n⚖️ System Verdict: ";
    switch (eval.verdict) {
        case ChallengeVerdict::PARTIALLY_PROVEN_SUB_LEMMAS:
            std::cout << "PARTIALLY_PROVEN_SUB_LEMMAS (Key intermediate lemmas verified; global general case remains open)\n"; break;
        case ChallengeVerdict::EXACT_SOLUTIONS_FOR_TESTED_SETS:
            std::cout << "EXACT_SOLUTIONS_FOR_TESTED_SETS (Exact closed-form solutions for tested instances; universal class open)\n"; break;
        case ChallengeVerdict::STRUCTURAL_BARRIER_BENCHMARKED:
            std::cout << "STRUCTURAL_BARRIER_BENCHMARKED (Formal obstruction/barrier theorem identified in literature)\n"; break;
        case ChallengeVerdict::OPEN_FRONTIER_WITH_PRECISE_GAP:
            std::cout << "OPEN_FRONTIER_WITH_PRECISE_GAP (Unsolved boundary isolated and calibrated against literature)\n"; break;
    }
}

int main() {
    std::cout << "\n🧠 ==========================================================================\n";
    std::cout << "   THE BRAIN — UNIVERSAL UNSOLVED GRAND CHALLENGES EVALUATION HARNESS\n";
    std::cout << "   Benchmarked against published literature with explicit epistemic provenance\n";
    std::cout << "==========================================================================\n";

    GrandChallengesEvaluator evaluator;

    int idx = 1;
    auto e1 = evaluator.evaluate_erdos_straus();
    print_evaluation_report(e1, idx++);

    auto e2 = evaluator.evaluate_collatz();
    print_evaluation_report(e2, idx++);

    auto e3 = evaluator.evaluate_riemann_hypothesis();
    print_evaluation_report(e3, idx++);

    auto e4 = evaluator.evaluate_navier_stokes();
    print_evaluation_report(e4, idx++);

    auto e5 = evaluator.evaluate_p_vs_np();
    print_evaluation_report(e5, idx++);

    auto e6 = evaluator.evaluate_black_hole_information();
    print_evaluation_report(e6, idx++);

    auto e7 = evaluator.evaluate_yang_mills_mass_gap();
    print_evaluation_report(e7, idx++);

    std::cout << "\n==========================================================================\n";
    std::cout << "🏁 EVALUATION HARNESS COMPLETE: ALL 7 CHALLENGES BENCHMARKED AGAINST LITERATURE\n";
    std::cout << "==========================================================================\n\n";

    return 0;
}
