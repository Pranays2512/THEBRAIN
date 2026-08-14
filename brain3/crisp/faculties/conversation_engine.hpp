#pragma once
#include <string>
#include <vector>
#include <map>
#include <set>
#include <deque>
#include <sstream>
#include "appraisal_engine.hpp"
#include "crisp/engines/reasoning/reasoning_engine.hpp"

namespace brain2 {
namespace faculties {

// Episodic Turn Buffer — lightweight short-term working memory for conversation
// Holds the last N turns (user query + brain response) so the Brain can reason
// across a multi-turn dialogue without the full EpisodicMemory overhead.
struct ConversationTurn {
    std::string role;    // "user" or "brain"
    std::string text;
    std::string topic;   // extracted entity focus of this turn
};

class ConversationEngine {
private:
    AppraisalEngine appraiser;
    std::string topic;                         // current working memory focus
    std::deque<ConversationTurn> turn_buffer;  // episodic short-term buffer
    static constexpr int MAX_TURNS = 10;       // keep last 10 turns (5 exchanges)

    reasoning::ReasoningEngine* kb = nullptr;  // pointer to shared knowledge base

    std::set<std::string> non_entity = {
        "what", "which", "how", "why", "who", "where", "when", "whose", "many", "much",
        "is", "are", "am", "do", "does", "did", "can", "could", "will", "would",
        "a", "an", "the", "of", "to", "i", "me", "my", "you", "your", "it", "that",
        "this", "in", "on", "for", "with", "and", "tell", "show", "describe",
        "explain", "give", "list", "about", "know", "please", "hey", "hi", "hello",
        "yes", "no", "not", "him", "her", "them", "they", "he", "she", "?"
    };

    std::string article(const std::string& word) {
        if (word.empty()) return "";
        char c = std::tolower(word[0]);
        if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') return "an";
        return "a";
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
        return toks;
    }

    std::vector<std::string> extract_entities(const std::string& text) {
        std::vector<std::string> ents;
        for (const auto& w : tokenize(text)) {
            if (!non_entity.count(w)) ents.push_back(w);
        }
        return ents;
    }

    // Detect if user is referring back to previous topic (pronoun reference resolution)
    std::string resolve_topic(const std::vector<std::string>& ents, const std::string& query) {
        if (!ents.empty()) return ents[0];
        // No entity found in current query — check if user said "it", "that", "this"
        // and resolve to the most recent topic from the turn buffer
        auto toks = tokenize(query);
        for (const auto& t : toks) {
            if (t == "it" || t == "that" || t == "this" || t == "him" || t == "her") {
                if (!turn_buffer.empty()) return turn_buffer.back().topic;
            }
        }
        return topic; // fall back to working memory focus
    }

    void push_turn(const std::string& role, const std::string& text, const std::string& t) {
        if (turn_buffer.size() >= MAX_TURNS) turn_buffer.pop_front();
        turn_buffer.push_back({role, text, t});
    }

    // Build a query response using the ReasoningEngine
    std::string query_kb(const std::string& subj, const std::string& rel) {
        if (!kb) return "";
        auto [ans, why] = kb->ask(subj, rel);
        if (!ans.empty()) return ans;
        return "";
    }

    std::string format_answer(const std::string& subj, const std::string& rel, const std::string& obj, const std::string& why) {
        if (obj.empty()) return "";
        std::string response = subj + " " + rel + " " + obj;
        if (!why.empty()) response += " [" + why + "]";
        return response;
    }

public:
    ConversationEngine() = default;

    // Attach a knowledge base so the engine can query real facts
    void set_knowledge_base(reasoning::ReasoningEngine& knowledge) {
        kb = &knowledge;
    }

    // Returns the episodic turn buffer for inspection (e.g. by the SleepEngine)
    const std::deque<ConversationTurn>& get_turns() const { return turn_buffer; }

    std::string respond(const std::string& text) {
        auto ap = appraiser.appraise(text);

        if (ap.type == "greeting") {
            std::string reply = "Hello! Ask me about something you've taught me.";
            push_turn("brain", reply, "");
            return reply;
        }

        auto ents = extract_entities(text);
        std::string resolved_topic = resolve_topic(ents, text);

        if (resolved_topic.empty()) {
            push_turn("user", text, "");
            std::string reply = "I'm not sure what you're asking about.";
            push_turn("brain", reply, "");
            return reply;
        }

        topic = resolved_topic;
        push_turn("user", text, topic);

        if (ap.type == "question" && kb) {
            // Try common relations: is_a, has, can, made_of, part_of, property
            std::vector<std::string> relations = {"is_a", "has", "can", "made_of", "part_of", "property", "color", "size", "speed"};
            std::vector<std::string> found_answers;

            for (const auto& rel : relations) {
                auto [ans, why] = kb->ask(topic, rel);
                if (!ans.empty() && ans != "<EXCEPTION>") {
                    found_answers.push_back(rel + ": " + ans);
                }
            }

            if (!found_answers.empty()) {
                std::ostringstream oss;
                oss << "About " << topic << " — ";
                for (size_t i = 0; i < found_answers.size(); ++i) {
                    oss << found_answers[i];
                    if (i + 1 < found_answers.size()) oss << "; ";
                }
                std::string reply = oss.str();
                push_turn("brain", reply, topic);
                return reply;
            }

            // Check episodic turn buffer for recent context on this topic
            for (auto it = turn_buffer.rbegin(); it != turn_buffer.rend(); ++it) {
                if (it->topic == topic && it->role == "brain") {
                    std::string reply = "From what I said earlier: " + it->text;
                    push_turn("brain", reply, topic);
                    return reply;
                }
            }

            std::string reply = "I don't have any information about " + topic + " yet.";
            push_turn("brain", reply, topic);
            return reply;
        }

        if (ap.type == "statement" && kb) {
            // Teach mode: "X is Y" or "X has Y" — extract and store as a fact
            auto toks = tokenize(text);
            // Simple SVO extractor: entity [is/has/can] entity
            for (size_t i = 1; i + 1 < toks.size(); ++i) {
                if (toks[i] == "is" || toks[i] == "has" || toks[i] == "can") {
                    std::string rel = (toks[i] == "is") ? "is_a" : toks[i];
                    std::string obj = toks[i + 1];
                    if (!non_entity.count(obj)) {
                        kb->learn(topic, rel, obj);
                        std::string reply = "Understood. I've learned that " + topic + " " + toks[i] + " " + obj + ".";
                        push_turn("brain", reply, topic);
                        return reply;
                    }
                }
            }
        }

        std::string reply = "Noted about " + topic + ".";
        push_turn("brain", reply, topic);
        return reply;
    }
};

} // namespace faculties
} // namespace brain2
