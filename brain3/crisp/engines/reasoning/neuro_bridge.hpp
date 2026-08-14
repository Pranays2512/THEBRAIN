#pragma once
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <iostream>

#include "crisp/engines/reasoning/brainql.hpp"

namespace brain2 {
namespace reasoning {

struct Query {
    std::string kind;
    std::map<std::string, std::string> payload;
    std::string raw;
};

struct Answer {
    std::string kind;
    bool known;
    bool verified;
    std::string value;
    std::vector<std::string> steps;
    std::string note;
};

class Eyes {
public:
    virtual ~Eyes() = default;
    virtual Query parse(const std::string& text) = 0;
};

class Mouth {
public:
    virtual ~Mouth() = default;
    virtual std::string render(const Answer& ans) = 0;
};

class Brain {
private:
    ReasoningEngine re;
public:
    Brain() {}
    ReasoningEngine* get_engine() { return &re; }

    void teach(const std::string& s, const std::string& rel, const std::string& o) {
        re.learn(s, rel, o);
    }

    Answer answer(const Query& q) {
        if (q.kind == "language") {
            auto text_in = q.payload.at("text");
            // simplified for C++ port verification
            auto res = re.ask(text_in, "isa");
            if (!res.first.empty()) {
                return {"language", true, false, res.first, {res.second}, ""};
            }
            return {"language", false, false, "I don't know", {}, "not in KB"};
        }
        return {"error", false, false, "", {}, "unknown query kind"};
    }
};

class RuleEyes : public Eyes {
public:
    Query parse(const std::string& text) override {
        // very simplified NLP rule parser
        return {"language", {{"text", text}}, text};
    }
};

class GrammarMouth : public Mouth {
public:
    std::string render(const Answer& ans) override {
        if (ans.kind == "language") return ans.value;
        return "I couldn't understand that.";
    }
};

class Mind {
private:
    std::shared_ptr<Eyes> eyes;
    std::shared_ptr<Brain> brain;
    std::shared_ptr<Mouth> mouth;
    BrainQLExecutor bql_exec;

public:
    Mind(std::shared_ptr<Eyes> e, std::shared_ptr<Brain> b, std::shared_ptr<Mouth> m)
        : eyes(e), brain(b), mouth(m), bql_exec(b->get_engine()) {}

    void teach(const std::string& s, const std::string& r, const std::string& o) {
        brain->teach(s, r, o);
    }

    std::string respond(const std::string& text) {
        if (text.find("INHERIT") == 0 || text.find("TEACH") == 0 || text.find("CHAIN") == 0) {
            try {
                auto bq = parse_bql(text);
                auto res = bql_exec.run(bq);
                if (res.known) return "BrainQL: " + res.value;
                return "BrainQL Unknown: " + res.note;
            } catch (const BrainQLParseError& e) {
                return std::string("BrainQL Error: ") + e.what();
            }
        }
        auto q = eyes->parse(text);
        auto ans = brain->answer(q);
        return mouth->render(ans);
    }
};

} // namespace reasoning
} // namespace brain2
