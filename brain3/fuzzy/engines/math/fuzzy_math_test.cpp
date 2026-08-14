#include <iostream>
#include <iomanip>
#include "prob_compute.hpp"

using namespace brain2::math;

void test_prob_lm() {
    std::cout << "=== prob_compute — N-gram sequence modeling ===\n\n";
    ProbLM lm(3);
    
    // Train on a tiny internal corpus
    std::vector<std::string> corpus = {
        "The object has high velocity",
        "The rocket has high velocity",
        "The object has large mass",
        "The rocket has large mass",
        "Physics is beautiful"
    };
    lm.train(corpus);
    
    std::cout << "trained an internal language model on " << corpus.size() << " sentences.\n\n";
    
    std::cout << "FORM SENTENCES:\n";
    for (int i = 0; i < 4; ++i) {
        auto sent = lm.generate({"<s>"}, 14, i);
        std::cout << "  -> ";
        for (const auto& w : sent) std::cout << w << " ";
        std::cout << "\n";
    }
    
    std::cout << "\nUNCERTAINTY:\n";
    std::vector<std::vector<std::string>> contexts = {
        {"the", "rocket", "has", "high"},
        {"the", "object", "has", "large"}
    };
    
    for (const auto& ctx : contexts) {
        auto d = lm.dist(ctx);
        float h = lm.entropy(ctx);
        std::cout << "  P(next | '...";
        if (ctx.size() >= 2) std::cout << ctx[ctx.size()-2] << " " << ctx[ctx.size()-1];
        std::cout << "') = { ";
        for (const auto& kv : d) std::cout << kv.first << ": " << std::setprecision(2) << kv.second << " ";
        std::cout << "} entropy " << h << " bits\n";
    }
}

int main() {
    test_prob_lm();
    return 0;
}
