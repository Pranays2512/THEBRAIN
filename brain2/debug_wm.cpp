#include "core/working_mem.hpp"
#include <iostream>

using namespace brain2;

int main() {
    WorkingMemory wm(16, 7, 0.95f);
    std::vector<float> vec(16, 5.0f);
    wm.gate(vec, 1.0f);
    wm.tick();
    auto ctx = wm.context();
    for(auto x : ctx) std::cout << x << " ";
    std::cout << "\n";
    return 0;
}
