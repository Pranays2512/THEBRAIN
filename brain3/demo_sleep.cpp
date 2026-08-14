#include <iostream>
#include <string>
#include "crisp/engines/reasoning/reasoning_engine.hpp"
#include "crisp/engines/reasoning/sleep_engine.hpp"

using namespace brain2::reasoning;

int main() {
    std::cout << "===========================================\n";
    std::cout << "  BRAIN 3: SLEEP / MEMORY CONSOLIDATION\n";
    std::cout << "===========================================\n";

    ReasoningEngine kb;
    
    std::cout << "1. Brain is awake. Observing 100 birds...\n";
    for (int i = 1; i <= 100; ++i) {
        std::string bird_id = "bird_" + std::to_string(i);
        kb.learn(bird_id, "is_a", "bird");
        kb.learn(bird_id, "can", "fly");
        kb.learn(bird_id, "has", "feathers");
    }

    std::cout << "2. Observing an exception: A penguin!\n";
    kb.learn("penguin_1", "is_a", "bird");
    kb.learn("penguin_1", "can", "swim");
    // Notice we do NOT explicitly say it can't fly, it just lacks the (can, fly) trait.
    // The Sleep engine will have to figure out how to prevent hallucination here.

    std::cout << "\n[Before Sleep] Total facts in memory: " << kb.facts.size() << "\n\n";

    std::cout << "3. The Brain goes to Sleep (Consolidating memories)...\n";
    SleepEngine sleep_engine;
    sleep_engine.sleep(kb, 0.90, 10); // 90% confidence threshold, min support 10

    std::cout << "\n[After Sleep] Total facts in memory: " << kb.facts.size() << "\n\n";

    std::cout << "4. Waking up! Testing Knowledge Retrieval (Default Logic)...\n";

    auto ask = [&](const std::string& subj, const std::string& rel) {
        auto [ans, reason] = kb.ask(subj, rel);
        if (ans.empty()) {
            std::cout << "  ? " << subj << " " << rel << " ? -> (I don't know)\n";
        } else {
            std::cout << "  > " << subj << " " << rel << " " << ans << "  [" << reason << "]\n";
        }
    };

    std::cout << "Querying a normal bird:\n";
    ask("bird_42", "can");
    ask("bird_42", "has");

    std::cout << "\nQuerying the exception (Penguin):\n";
    ask("penguin_1", "can");
    ask("penguin_1", "has"); // It should still inherit 'feathers' because penguins have feathers!

    return 0;
}
