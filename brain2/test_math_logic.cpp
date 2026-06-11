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

    std::cout << "Writing '5' and '3' to Scratchpad..." << std::endl;
    b.scratchpad.write("subject", b.language.encode("5"), "math");
    b.scratchpad.write("object", b.language.encode("3"), "math");

    std::cout << "Executing Op::MATH_ADD (5 + 3)..." << std::endl;
    b.logic_engine.execute_op(Op::MATH_ADD, b.scratchpad);

    auto res = b.scratchpad.read("result");
    std::string res_word = b.language.best_word(res, {}, 0);
    std::cout << "Result Word: " << res_word << std::endl;

    std::cout << "Executing Op::MATH_SUB (5 - 3)..." << std::endl;
    b.logic_engine.execute_op(Op::MATH_SUB, b.scratchpad);

    res = b.scratchpad.read("result");
    res_word = b.language.best_word(res, {}, 0);
    std::cout << "Result Word: " << res_word << std::endl;

    std::cout << "Writing '10' and '2' to Scratchpad..." << std::endl;
    b.scratchpad.write("result", b.language.encode("10"), "math"); // MATH_DIV uses result and relation
    b.scratchpad.write("relation", b.language.encode("2"), "math");

    std::cout << "Executing Op::MATH_DIV (10 / 2)..." << std::endl;
    b.logic_engine.execute_op(Op::MATH_DIV, b.scratchpad);

    res = b.scratchpad.read("result");
    res_word = b.language.best_word(res, {}, 0);
    std::cout << "Result Word: " << res_word << std::endl;

    return 0;
}
