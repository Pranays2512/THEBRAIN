#include <iostream>
#include <vector>
#include "code_engine.hpp"

using namespace brain2;

int main() {
    std::cout << "======================================================================\n";
    std::cout << "  CodeEngine (C++ Port) - Neural-Guided A* Search \n";
    std::cout << "======================================================================\n\n";

    CodeEngine engine;

    // Test 1: Contains Duplicate
    // io: ([1, 2, 3, 1], True), ([1, 2, 3, 4], False)
    {
        std::cout << "[Problem: Contains Duplicate]\n";
        std::vector<int> in1 = {1, 2, 3, 1}; bool out1 = true;
        std::vector<int> in2 = {1, 2, 3, 4}; bool out2 = false;
        
        std::vector<std::pair<State, State>> io = {
            {State(in1), State(out1)},
            {State(in2), State(out2)}
        };

        auto res = engine.synthesize(io);
        if (res.tree) {
            std::cout << "  [Discovered Python Code]:\n" << res.code << "\n";
        } else {
            std::cout << "  FAILED to synthesize.\n";
        }
    }

    // Test 2: Sort and double positive numbers
    // io: ([3, -1, 2], [4, 6]), ([0, 5, -5, 1], [2, 10])
    {
        std::cout << "\n[Problem: Sort and double the positive numbers]\n";
        std::vector<int> in1 = {3, -1, 2}; std::vector<int> out1 = {4, 6};
        std::vector<int> in2 = {0, 5, -5, 1}; std::vector<int> out2 = {2, 10};
        
        std::vector<std::pair<State, State>> io = {
            {State(in1), State(out1)},
            {State(in2), State(out2)}
        };

        auto res = engine.synthesize(io);
        if (res.tree) {
            std::cout << "  [Discovered C++ Code]:\n" << res.code_cpp << "\n";
        } else {
            std::cout << "  FAILED to synthesize.\n";
        }
    }

    // Test 3: Two Sum (Target = 9)
    // io: ([2, 7, 11, 15], [0, 1]), ([3, 2, 4, 7], [1, 3])
    {
        std::cout << "\n[Problem: Two Sum (Target = 9)]\n";
        engine.target_val = 9;
        std::vector<int> in1 = {2, 7, 11, 15}; std::vector<int> out1 = {0, 1};
        std::vector<int> in2 = {3, 2, 4, 7}; std::vector<int> out2 = {1, 3};
        
        std::vector<std::pair<State, State>> io = {
            {State(in1), State(out1)},
            {State(in2), State(out2)}
        };

        auto res = engine.synthesize(io, 5000);
        if (res.tree) {
            std::cout << "  [Discovered Python Code]:\n" << res.code << "\n";
        } else {
            std::cout << "  FAILED to synthesize.\n";
        }
    }

    return 0;
}
