#pragma once
#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

namespace brain2 {
namespace store {

// A lightweight serialization store, meant to accumulate knowledge
// across sessions instead of starting from scratch.
class BrainStore {
public:
    std::string path;
    
    // Knowledge stores
    std::map<std::string, std::string> facts;
    std::map<std::string, std::string> policies;
    std::map<std::string, std::string> functions;
    
    BrainStore(const std::string& p) : path(p) {
        load();
    }
    
    void load() {
        _load_map(path + "/facts.txt", facts);
        _load_map(path + "/policies.txt", policies);
        _load_map(path + "/functions.txt", functions);
    }
    
    void save() const {
        _save_map(path + "/facts.txt", facts);
        _save_map(path + "/policies.txt", policies);
        _save_map(path + "/functions.txt", functions);
    }
    
    bool knows_fact(const std::string& k) const { return facts.count(k); }
    bool knows_policy(const std::string& k) const { return policies.count(k); }
    bool knows_function(const std::string& k) const { return functions.count(k); }
    
    void add_fact(const std::string& k, const std::string& v) { facts[k] = v; }
    void add_policy(const std::string& k, const std::string& v) { policies[k] = v; }
    void add_function(const std::string& k, const std::string& v) { functions[k] = v; }
    
    std::string summary() const {
        return std::to_string(policies.size()) + " policies, " +
               std::to_string(functions.size()) + " functions, " +
               std::to_string(facts.size()) + " facts";
    }

private:
    void _load_map(const std::string& file_path, std::map<std::string, std::string>& m) {
        std::ifstream f(file_path);
        if (!f.is_open()) return;
        std::string line;
        while (std::getline(f, line)) {
            auto delim = line.find("=");
            if (delim != std::string::npos) {
                m[line.substr(0, delim)] = line.substr(delim + 1);
            }
        }
    }
    
    void _save_map(const std::string& file_path, const std::map<std::string, std::string>& m) const {
        std::ofstream f(file_path);
        if (!f.is_open()) return;
        for (const auto& [k, v] : m) {
            f << k << "=" << v << "\n";
        }
    }
};

} // namespace store
} // namespace brain2
