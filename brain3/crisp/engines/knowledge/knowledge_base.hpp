#pragma once
#include <vector>
#include <string>
#include <map>
#include <set>
#include <tuple>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "fact_extractor.hpp"

namespace brain2 {
namespace knowledge {

class KnowledgeBase {
public:
    std::set<std::tuple<std::string, std::string, std::string>> facts;
    std::map<std::string, int> by_source;
    
    std::string norm(const std::string& token) {
        std::string t = token;
        t.erase(t.find_last_not_of(" \n\r\t") + 1);
        t.erase(0, t.find_first_not_of(" \n\r\t"));
        std::transform(t.begin(), t.end(), t.begin(), ::tolower);
        std::replace(t.begin(), t.end(), ' ', '_');
        return t;
    }
    
    bool add(const std::string& s_raw, const std::string& r_raw, const std::string& o_raw, const std::string& source = "manual") {
        std::string s = norm(s_raw);
        std::string r = norm(r_raw);
        std::string o = norm(o_raw);
        if (s.empty() || r.empty() || o.empty() || s == o) return false;
        
        auto tup = std::make_tuple(s, r, o);
        if (facts.count(tup)) return false;
        
        facts.insert(tup);
        by_source[source]++;
        return true;
    }
    
    int ingest_triples(const std::vector<std::tuple<std::string, std::string, std::string>>& triples, const std::string& source) {
        int n = 0;
        for (const auto& t : triples) {
            if (add(std::get<0>(t), std::get<1>(t), std::get<2>(t), source)) n++;
        }
        return n;
    }
    
    int ingest_text(const std::string& text, const std::string& source = "text") {
        FactExtractor ex;
        return ingest_triples(ex.extract(text), source);
    }
    
    int ingest_tsv(const std::string& path, const std::string& source = "tsv") {
        int n = 0;
        std::ifstream file(path);
        std::string line;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string s, r, o;
            if (std::getline(ss, s, '\t') && std::getline(ss, r, '\t') && std::getline(ss, o, '\t')) {
                if (add(s, r, o, source)) n++;
            }
        }
        return n;
    }
    
    int stats_facts() const { return facts.size(); }
    int stats_entities() const {
        std::set<std::string> ents;
        for (const auto& t : facts) {
            ents.insert(std::get<0>(t));
            ents.insert(std::get<2>(t));
        }
        return ents.size();
    }
    int stats_relations() const {
        std::set<std::string> rels;
        for (const auto& t : facts) rels.insert(std::get<1>(t));
        return rels.size();
    }
};

} // namespace knowledge
} // namespace brain2
