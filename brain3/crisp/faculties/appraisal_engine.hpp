#pragma once
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <sstream>

namespace brain2 {
namespace faculties {

struct Appraisal {
    std::map<std::string, double> frame;
    std::string type;
};

class AppraisalEngine {
private:
    std::vector<std::string> dimensions = {
        "question", "greeting", "command", "curious", "friendly",
        "polite", "about_self", "definition", "negation"
    };

    std::map<std::string, std::map<std::string, double>> markers = {
        {"what", {{"question", 1.0}, {"definition", 1.0}}},
        {"which", {{"question", 1.0}, {"definition", 1.0}}},
        {"how", {{"question", 1.0}, {"curious", 1.0}}},
        {"why", {{"question", 1.0}, {"curious", 1.0}}},
        {"who", {{"question", 1.0}}}, {"where", {{"question", 1.0}}},
        {"when", {{"question", 1.0}}}, {"whose", {{"question", 1.0}}},
        {"?", {{"question", 1.0}}},
        {"do", {{"question", 0.5}}}, {"does", {{"question", 0.5}}},
        {"is", {{"question", 0.3}}}, {"are", {{"question", 0.3}}},
        {"can", {{"question", 0.5}}}, {"could", {{"question", 0.5}, {"polite", 1.0}}},
        {"will", {{"question", 0.4}}}, {"would", {{"question", 0.4}, {"polite", 1.0}}},
        {"hi", {{"greeting", 1.0}, {"friendly", 1.0}}},
        {"hello", {{"greeting", 1.0}, {"friendly", 1.0}}},
        {"thanks", {{"friendly", 1.0}}}, {"please", {{"polite", 1.0}}},
        {"you", {{"about_self", 1.0}}}, {"your", {{"about_self", 1.0}}},
        {"tell", {{"command", 1.0}}}, {"show", {{"command", 1.0}}},
        {"describe", {{"command", 1.0}}}, {"explain", {{"command", 1.0}}},
        {"not", {{"negation", 1.0}}}, {"no", {{"negation", 1.0}}}
    };

    std::vector<std::string> common_words = {
        "a", "an", "the", "of", "to", "i", "it", "and", "that", "this",
        "in", "on", "for", "with", "me", "my", "be", "am", "was", "were"
    };

    double get_weight(const std::string& token) {
        for (const auto& cw : common_words) if (cw == token) return 0.3;
        return 1.0;
    }

    std::vector<std::string> tokenize(std::string text) {
        for (auto& c : text) c = std::tolower(c);
        std::vector<std::string> toks;
        std::string current;
        for (char c : text) {
            if (std::isalpha(c) || c == '\'') current += c;
            else if (!current.empty()) { toks.push_back(current); current = ""; }
        }
        if (!current.empty()) toks.push_back(current);
        if (text.find('?') != std::string::npos) toks.push_back("?");
        return toks;
    }

public:
    Appraisal appraise(const std::string& text) {
        std::map<std::string, double> frame;
        for (const auto& d : dimensions) frame[d] = 0.0;
        
        for (const auto& tok : tokenize(text)) {
            if (markers.count(tok)) {
                double w = get_weight(tok);
                for (const auto& [dim, val] : markers[tok]) {
                    frame[dim] += val * w;
                }
            }
        }
        
        std::string type = "statement";
        if (frame["question"] >= 0.8) type = "question";
        else if (frame["greeting"] >= 1.0 && frame["question"] < 0.8) type = "greeting";
        else if (frame["command"] >= 1.0) type = "command";
        else if (frame["question"] >= 0.4) type = "question";
        
        return {frame, type};
    }
};

} // namespace faculties
} // namespace brain2
