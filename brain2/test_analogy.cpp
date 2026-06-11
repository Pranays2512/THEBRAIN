#include <iostream>
#include <string>
#include <vector>
#include "core/brain.hpp"

using namespace brain2;

int main() {
    std::cout << "Loading Brain..." << std::endl;
    Brain b(256, 256, 128, 256);
    std::string cd = "checkpoints/math_brain";
    b.load_components(
        cd + "/predictor.bin", cd + "/language.bin", cd + "/som.bin", 
        cd + "/episodic.bin", cd + "/emotion.bin", cd + "/self.bin",
        cd + "/symbolic.bin", cd + "/binding.bin", cd + "/bg.bin",
        cd + "/procedures.bin", cd + "/hpred.bin"
    );

    std::cout << "Injecting a structural binding memory: 'Bird' 'Has' 'Feathers'" << std::endl;
    auto bird = b.language.encode("bird");
    auto has  = b.language.encode("has");
    auto feathers = b.language.encode("feathers");
    b.binding.bind(bird, has, feathers);

    std::cout << "Writing novel scenario to Scratchpad: 'Dog' 'Has' -> [Context: 'Bird']" << std::endl;
    auto dog = b.language.encode("dog");
    
    // We want the Brain to infer what a Dog has, by analogy to the Bird.
    b.scratchpad.write("subject", dog, "analogy");
    b.scratchpad.write("relation", has, "analogy");
    b.scratchpad.write("context_map", bird, "analogy"); // Context is bird, to prompt structural mapping

    std::cout << "Executing Op::ANALOGY..." << std::endl;
    b.logic_engine.execute_op(Op::ANALOGY, b.scratchpad);

    auto res = b.scratchpad.read("result");
    if (!res.empty()) {
        std::string res_word = b.language.best_word(res, {}, 0);
        std::cout << "Analogy Result Word: " << res_word << std::endl;
        if (res_word == "feathers") {
            std::cout << "SUCCESS: The Brain structurally mapped 'Dog' to 'Bird' and inferred 'feathers'!" << std::endl;
        } else {
            std::cout << "FAILED: The Brain did not infer the structural analogy." << std::endl;
        }
    } else {
        std::cout << "FAILED: No result written to scratchpad." << std::endl;
    }

    return 0;
}
