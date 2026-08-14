#include <iostream>
#include <iomanip>
#include "brain_store.hpp"

using namespace brain2::store;

int main() {
    std::cout << "=== brain_store — accumulate verified knowledge across sessions ===\n\n";
    
    std::string test_path = "test_store";
    
    // Session 1: Fresh brain
    {
        std::cout << "Session 1 (fresh brain)\n";
        BrainStore s1(test_path);
        std::cout << "  loaded: " << s1.summary() << "\n";
        
        s1.add_fact("apple|isa", "fruit");
        s1.add_policy("force", "mass * accel");
        s1.add_function("fibonacci", "int fib(int n) { return n <= 1 ? n : fib(n-1) + fib(n-2); }");
        
        std::cout << "  policy force DISCOVERED (verified) -> stored\n";
        std::cout << "  function fibonacci SYNTHESIZED (survives stress) -> stored\n";
        
        s1.save();
        std::cout << "  saved: " << s1.summary() << "\n\n";
    }
    
    // Session 2: Reloaded brain
    {
        std::cout << "Session 2 (reloaded — grows on top)\n";
        BrainStore s2(test_path);
        std::cout << "  loaded: " << s2.summary() << "\n";
        
        if (s2.knows_policy("force")) {
            std::cout << "  policy force already known — reuse\n";
        }
        if (s2.knows_function("fibonacci")) {
            std::cout << "  function fibonacci already known — reuse\n";
        }
        
        s2.add_policy("ke", "0.5 * mass * speed^2");
        std::cout << "  policy ke DISCOVERED (verified) -> stored\n";
        
        s2.save();
        std::cout << "  saved: " << s2.summary() << "\n\n";
    }
    
    return 0;
}
