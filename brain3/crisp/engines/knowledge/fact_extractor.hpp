#pragma once
#include <vector>
#include <string>
#include <regex>
#include <set>
#include <tuple>
#include <iostream>

namespace brain2 {
namespace knowledge {

const std::set<std::string> ARTICLES = {"a", "an", "the"};
const std::set<std::string> PRONOUNS = {"it", "they", "he", "she", "this", "that"};

// Very basic stemming for simple relations
inline std::string stem(const std::string& verb) {
    std::string v = verb;
    if (v.length() > 2 && v.substr(v.length()-2) == "es") return v.substr(0, v.length()-2);
    if (v.length() > 1 && v.substr(v.length()-1) == "s") return v.substr(0, v.length()-1);
    return v;
}

struct FactExtractor {
    std::vector<std::pair<std::regex, std::string>> patterns;
    
    FactExtractor() {
        patterns = {
            {std::regex(R"(^([\w\s]+?) is (?:the |a |an )?(\w+) of (\w+)$)"), "%MID%"},
            {std::regex(R"(^([\w\s]+?) is an? ([\w\s]+)$)"), "isa"},      // multiword subj+obj
            {std::regex(R"(^([\w\s]+?) are an? ([\w\s]+)$)"), "isa"},
            {std::regex(R"(^(\w+) is an? (\w+)$)"), "isa"},
            {std::regex(R"(^(\w+) are an? (\w+)$)"), "isa"},
            {std::regex(R"(^(\w+) (?:is|are) (\w+)$)"), "is"},
            {std::regex(R"(^(\w+) (?:has|have) an? (\w+)$)"), "has"},
            {std::regex(R"(^(\w+) (?:has|have) (\w+)$)"), "has"},
            {std::regex(R"(^(\w+) (?:grows?|grow) on (?:an? )?(\w+)$)"), "grows_on"},
            {std::regex(R"(^(\w+) (?:lives?|live) in (?:an? )?(\w+)$)"), "lives_in"},
            {std::regex(R"(^(\w+) (?:gives?|give) (?:an? )?(\w+)$)"), "gives"},
            {std::regex(R"(^(\w+) (?:eats?|eat) (?:an? )?(\w+)$)"), "eats"},
            {std::regex(R"(^(\w+) (\w+) an? (\w+)$)"), ""}, // generic
            {std::regex(R"(^(\w+) (\w+) (\w+)$)"), ""} // generic
        };
    }
    
    std::vector<std::string> get_sentences(const std::string& text) {
        std::vector<std::string> sents;
        std::regex sep(R"([.!?\n]+)");
        std::sregex_token_iterator it(text.begin(), text.end(), sep, -1);
        std::sregex_token_iterator end;
        for (; it != end; ++it) {
            std::string s = *it;
            s.erase(0, s.find_first_not_of(" \t\r\n"));
            s.erase(s.find_last_not_of(" \t\r\n") + 1);
            if (!s.empty()) sents.push_back(s);
        }
        return sents;
    }
    
    std::vector<std::string> clean_words(const std::string& sent) {
        std::string s = sent;
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        std::regex word_re(R"([a-z\']+)");
        std::vector<std::string> words;
        std::sregex_token_iterator it(s.begin(), s.end(), word_re);
        std::sregex_token_iterator end;
        for (; it != end; ++it) words.push_back(*it);
        
        if (!words.empty() && ARTICLES.count(words[0])) {
            words.erase(words.begin());
        }
        return words;
    }
    
    std::pair<std::optional<std::tuple<std::string, std::string, std::string>>, std::string> 
    extract_one(const std::string& sent, const std::string& last_subj) {
        auto words = clean_words(sent);
        if (words.empty()) return {std::nullopt, last_subj};
        
        if (PRONOUNS.count(words[0])) {
            if (last_subj.empty()) return {std::nullopt, last_subj};
            words[0] = last_subj;
        }
        
        std::string joined = "";
        for (size_t i = 0; i < words.size(); ++i) {
            joined += words[i] + (i == words.size()-1 ? "" : " ");
        }
        
        for (const auto& pat_pair : patterns) {
            std::smatch m;
            if (std::regex_match(joined, m, pat_pair.first)) {
                std::string subj, rel, obj;
                if (pat_pair.second == "%MID%") {
                    subj = m[1]; rel = m[2]; obj = m[3];
                } else if (pat_pair.second == "") {
                    subj = m[1]; std::string verb = m[2]; obj = m[3];
                    if (ARTICLES.count(verb) || ARTICLES.count(obj)) continue;
                    rel = stem(verb);
                } else {
                    subj = m[1]; obj = m[2];
                    rel = pat_pair.second;
                }
                
                if (ARTICLES.count(subj) || ARTICLES.count(obj) || subj == obj) continue;
                return {std::make_tuple(subj, rel, obj), subj};
            }
        }
        return {std::nullopt, last_subj};
    }
    
    std::vector<std::tuple<std::string, std::string, std::string>> extract(const std::string& text) {
        std::vector<std::tuple<std::string, std::string, std::string>> triples;
        std::string last_subj = "";
        for (const auto& sent : get_sentences(text)) {
            auto res = extract_one(sent, last_subj);
            if (res.first.has_value()) {
                triples.push_back(*res.first);
            }
            last_subj = res.second;
        }
        return triples;
    }
};

} // namespace knowledge
} // namespace brain2
