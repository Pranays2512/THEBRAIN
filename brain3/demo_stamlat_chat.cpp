// demo_stamlat_chat.cpp — STAMLAT v2 conversation training + interactive REPL
// usage: demo_stamlat_chat [--quick] [--steps N]
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "crisp/engines/neural/stamlat_transformer.hpp"

using namespace brain3::engines::neural;

// Reference-to-array size helper — a bare `auto` lambda parameter decays to
// a pointer, which made sizeof/sizeof == 1 and every pick index 0.
template <typename T, size_t N>
constexpr size_t n_of(const T (&)[N]) { return N; }

static const char* kFallbackCorpus =
    "user: hello\nbrain: intent greeting style friendly\n"
    "user: hi\nbrain: intent greeting emotion positive\n"
    "user: what is your name\nbrain: name brain type ai\n"
    "user: who are you\nbrain: identity system type cognitive\n"
    "user: what do you do\nbrain: action learning goal communication\n"
    "user: how are you\nbrain: status good energy high\n";

static std::string load_conversations(const std::string& path) {
    std::ifstream f(path);
    if (!f) return kFallbackCorpus;
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string raw = ss.str();
    std::string out;
    std::istringstream lines(raw);
    std::string line;
    while (std::getline(lines, line)) {
        for (auto& c : line) c = (char)std::tolower((unsigned char)c);
        out += "user: " + line + "\nbrain: ";
        // split the templated answer: corpus lines are "<question> <answer slots>"
        // answer = last 4 words (slot pattern), question = the rest
        std::vector<std::string> w;
        std::istringstream ws(line);
        std::string tok;
        while (ws >> tok) w.push_back(tok);
        if (w.size() > 4) {
            for (size_t i = 0; i + 4 < w.size(); ++i) { /* consumed in prefix */ }
        }
        out += "\n";
    }
    return out;
}

int main(int argc, char** argv) {
    bool quick = false;
    int steps = 6000;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--quick") quick = true;
        else if (a == "--steps" && i + 1 < argc) steps = std::atoi(argv[++i]);
    }
    if (quick && steps == 6000) steps = 1200;

    StamlatConfig cfg;
    cfg.d_model = 96; cfg.n_layers = 3; cfg.n_heads = 6; cfg.d_ff = 256;
    cfg.ctx = 48; cfg.depth_gamma = 0.0f; cfg.depth_tau = 1.0f; cfg.seed = 42;
    StamlatLM lm(cfg);

    // Build dialogue-formatted training text directly from the generator's templates.
    static const char* Q[] = {"how are you","how is your day","how are you doing","what is up","how goes it"};
    static const char* QA[] = {"status optimal emotion happy","status good energy high","state positive mode ready","feeling great condition excellent"};
    static const char* W[] = {"who are you","what is your name","what are you","tell me about yourself"};
    static const char* WA[] = {"identity system type cognitive","identity brain origin artificial","self network type neural","name brain type ai"};
    static const char* D[] = {"what are you doing","what do you do","what is your purpose"};
    static const char* DA[] = {"action learning goal communication","task processing goal understanding","purpose learning objective thinking","activity reading state processing"};
    static const char* G[] = {"hello","hi","hey there","greetings","good morning","good evening"};
    static const char* GA[] = {"intent greeting style friendly","intent salutation status ready","intent greeting emotion positive","intent welcome target user"};

    std::mt19937 rng(7);
    auto pick = [&](const auto& arr) { return arr[rng() % n_of(arr)]; };

    std::string train, heldout;
    for (int i = 0; i < 1500; ++i) {
        switch (rng() % 4) {
            case 0: train += std::string("user: ") + pick(Q) + "\nbrain: " + pick(QA) + "\n"; break;
            case 1: train += std::string("user: ") + pick(W) + "\nbrain: " + pick(WA) + "\n"; break;
            case 2: train += std::string("user: ") + pick(D) + "\nbrain: " + pick(DA) + "\n"; break;
            default: train += std::string("user: ") + pick(G) + "\nbrain: " + pick(GA) + "\n"; break;
        }
    }
    // held-out generalization set: phrasings never seen in training prompts
    for (int i = 0; i < 60; ++i)
        heldout += std::string("user: hey how are you feeling\nbrain: ") + pick(QA) + "\n";
    for (int i = 0; i < 60; ++i)
        heldout += std::string("user: what are you called\nbrain: ") + pick(WA) + "\n";

    std::cout << "== STAMLAT v2 chat trainer ==\n";
    std::cout << "params: " << lm.param_count() << " | vocab: chars | ctx: " << cfg.ctx << "\n";
    lm.build_vocab(train);

    const float l0 = lm.eval_loss(train);
    std::cout << "initial eval loss: " << l0 << "\n";
    lm.fit(train, steps, 4e-3f, 12, steps / 8);
    const float l1 = lm.eval_loss(train);
    std::cout << "final eval loss:   " << l1 << "\n";

    std::cout << "\n-- sample turns --\n";
    const char* probes[] = {"hello", "who are you", "what do you do", "how goes it", "good evening"};
    for (const char* p : probes) {
        std::string reply = lm.complete(std::string("user: ") + p + "\nbrain: ", 48, 0.f);
        std::cout << "user: " << p << "\nbrain: " << reply << "\n";
    }
    std::cout << "\nheld-out phrasing loss: " << lm.eval_loss(heldout) << "\n";
    std::cout << "held-out turn: user: what are you called -> brain: "
              << lm.complete("user: what are you called\nbrain: ", 48, 0.f) << "\n";

    lm.save("stamlat_chat.bin");
    std::cout << "\nsaved model to stamlat_chat.bin\n";
    std::cout << "(interactive mode: pipe stdin lines, e.g. echo \"hello\" | ./demo_stamlat_chat)\n";

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) break;
        for (auto& c : line) c = (char)std::tolower((unsigned char)c);
        std::cout << "brain: " << lm.complete("user: " + line + "\nbrain: ", 48, 0.f) << "\n";
    }
    return 0;
}
