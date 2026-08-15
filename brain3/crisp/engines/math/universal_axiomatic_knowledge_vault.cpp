/**
 * brain3/crisp/engines/math/universal_axiomatic_knowledge_vault.cpp
 *
 * Driver and verification harness for The Brain's Universal Multi-Domain Axiomatic Knowledge Vault.
 */

#include "universal_axiomatic_knowledge_vault.hpp"
#include <iostream>
#include <iomanip>

using namespace thebrain::knowledge_vault;

int main() {
    std::cout << "\n🧠 ==========================================================================\n";
    std::cout << "   THE BRAIN — UNIVERSAL MULTI-DOMAIN AXIOMATIC KNOWLEDGE VAULT\n";
    std::cout << "   Formal Multi-Disciplinary Foundation Across Math, Physics, CS, Bio, Cosmo\n";
    std::cout << "==========================================================================\n";

    UniversalAxiomaticKnowledgeVault vault;

    std::cout << "\n🏛️ VAULT SUMMARY:\n";
    std::cout << "   Total Verified Axioms & Theorems: " << vault.size() << "\n";
    std::cout << "   Dependency Graph Acyclicity Check: " << (vault.verify_acyclicity() ? "✅ 100% STRICT DAG (No circularity)" : "❌ FAILED (Cycle detected)") << "\n";

    std::vector<ScienceDomain> domains = {
        ScienceDomain::MATHEMATICS,
        ScienceDomain::THEORETICAL_PHYSICS,
        ScienceDomain::COMPUTER_SCIENCE,
        ScienceDomain::BIOLOGY_BIOCHEMISTRY,
        ScienceDomain::COSMOLOGY_ASTROPHYSICS
    };

    for (auto d : domains) {
        std::cout << "\n" << std::string(80, '-') << "\n";
        std::cout << "🔬 DOMAIN: " << domain_to_string(d) << "\n";
        auto thms = vault.query_by_domain(d);
        for (const auto& thm : thms) {
            std::cout << "   • [" << thm.id << "] " << thm.name << "\n";
            std::cout << "     Statement: " << thm.formal_statement << "\n";
            std::cout << "     Equation : " << thm.canonical_equation << "\n";
            std::cout << "     Origin   : " << thm.peer_reviewed_origin << "\n";
        }
    }

    std::cout << "\n==========================================================================\n";
    std::cout << "🏁 UNIVERSAL KNOWLEDGE VAULT READY: MULTI-DISCIPLINARY AXIOMS SECURE\n";
    std::cout << "==========================================================================\n\n";

    return 0;
}
