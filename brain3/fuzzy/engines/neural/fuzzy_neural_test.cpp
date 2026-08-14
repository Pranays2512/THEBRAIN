#include <iostream>
#include <iomanip>
#include "neural_lm.hpp"

using namespace brain2::neural;

int main() {
    std::cout << "=== neural_lm — pure C++ dense sequence model ===\n\n";
    
    // We train on a tiny toy corpus just to prove the forward/backward logic 
    // runs, learns, and generates properly without crashing.
    std::vector<std::string> corpus = {
        "The object has high velocity",
        "The rocket has high velocity",
        "The object has large mass",
        "The rocket has large mass"
    };
    
    // k=2, d=8, h=16, lr=0.5, epochs=200 for a very quick toy test
    NeuralLM lm(2, 8, 16, 0.5f, 200);
    lm.train(corpus, 42);
    
    std::cout << "trained a native C++ neural LM on " << corpus.size() << " sentences (no libtorch/numpy).\n\n";
    
    std::cout << "GENERATE (sampled from the neural model):\n";
    for (int i = 0; i < 3; ++i) {
        auto sent = lm.generate(12, i);
        std::cout << "  -> ";
        for (const auto& w : sent) std::cout << w << " ";
        std::cout << "\n";
    }
    
    std::cout << "\nPREDICT on context:\n";
    std::vector<std::vector<std::string>> contexts = {
        {"the", "rocket", "has", "high"},
        {"the", "object", "has", "large"}
    };
    
    for (const auto& ctx : contexts) {
        auto d = lm.dist(ctx);
        std::cout << "  '...";
        if (ctx.size() >= 2) std::cout << ctx[ctx.size()-2] << " " << ctx[ctx.size()-1];
        std::cout << "' -> { ";
        
        // Print top 2 predictions
        std::vector<std::pair<float, std::string>> sorted_d;
        for (const auto& kv : d) {
            if (kv.first != "<s>" && kv.first != "</s>" && kv.first != "<unk>") {
                sorted_d.push_back({kv.second, kv.first});
            }
        }
        std::sort(sorted_d.rbegin(), sorted_d.rend());
        
        for (size_t i = 0; i < std::min(size_t(2), sorted_d.size()); ++i) {
            std::cout << sorted_d[i].second << ": " << std::setprecision(2) << sorted_d[i].first << " ";
        }
        std::cout << "}\n";
    }
    
    return 0;
}
