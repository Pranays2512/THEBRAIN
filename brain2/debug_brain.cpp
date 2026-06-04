#include "core/brain.hpp"
#include <iostream>
using namespace brain2;
int main() {
    Brain b(4, 4, 16);
    Language& l = b.language;
    l.register_word("nodeA");
    auto vec = l.encode("nodeA");
    for(auto& x : vec) x *= 10.0f;
    
    std::cout << "Initial pc_wm pred size: " << b.pc_wm.prediction.size() << "\n";
    
    for(int i=0; i<1; i++) {
        b.working_mem.gate(vec, 1.0f);
        b.think();
    }
    
    std::cout << "Prediction[0]: " << b.pc_wm.prediction[0] << "\n";
    std::cout << "Prediction[1]: " << b.pc_wm.prediction[1] << "\n";
    
    return 0;
}
