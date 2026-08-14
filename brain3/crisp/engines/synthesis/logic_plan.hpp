#pragma once
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <sstream>

namespace brain3 {
namespace engines {
namespace synthesis {

// ── LogicPlan: structured language-independent algorithm description ──────────
struct LogicPlan {
    std::string name;
    std::string description;
    std::vector<std::pair<std::string, std::string>> inputs;   // (name, type_hint)
    std::pair<std::string, std::string> output;                 // (name, type_hint)
    std::vector<std::string> steps;
    std::vector<std::string> data_structs;
    std::vector<std::string> invariants;
    std::map<std::string, std::string> complexity;
    std::vector<std::pair<std::map<std::string, std::string>, std::string>> test_cases; // (args, expected)

    std::string to_prompt(const std::string& lang = "cpp") const {
        std::ostringstream out;
        out << "Implement the following algorithm in " << lang << ".\n";
        out << "Algorithm: " << name << "\n";
        out << "Description: " << description << "\n\n";
        out << "Inputs:\n";
        for (const auto& [n, t] : inputs)
            out << "  - " << n << ": " << t << "\n";
        out << "Output: " << output.first << " (" << output.second << ")\n";

        if (!data_structs.empty()) {
            out << "\nData structures to use:\n";
            for (const auto& ds : data_structs)
                out << "  - " << ds << "\n";
        }

        out << "\nAlgorithm steps (implement EXACTLY in this order):\n";
        for (size_t i = 0; i < steps.size(); i++)
            out << "  " << (i + 1) << ". " << steps[i] << "\n";

        if (!invariants.empty()) {
            out << "\nInvariants (must hold throughout):\n";
            for (const auto& inv : invariants)
                out << "  - " << inv << "\n";
        }

        auto ti = complexity.find("time");
        auto si = complexity.find("space");
        if (ti != complexity.end() || si != complexity.end()) {
            out << "\nExpected complexity: ";
            if (ti != complexity.end()) out << "time=" << ti->second << " ";
            if (si != complexity.end()) out << "space=" << si->second;
            out << "\n";
        }

        out << "\nReturn ONLY the function named `" << name << "` with no extra explanation.\n";
        out << "Do not add imports. Do not change the function signature.\n";
        return out.str();
    }
};

// ── Plan registry: the Brain's library of structured algorithm blueprints ─────
inline std::map<std::string, LogicPlan> build_plan_registry() {
    std::map<std::string, LogicPlan> reg;

    reg["binary_search"] = {
        "binary_search",
        "Search for target in sorted array using divide-and-conquer",
        {{"arr", "vector<int>"}, {"target", "int"}},
        {"index", "int"},
        {
            "Initialize lo=0, hi=len(arr)-1",
            "While lo <= hi: mid = (lo+hi)//2",
            "If arr[mid]==target return mid",
            "If arr[mid]<target set lo=mid+1 else hi=mid-1",
            "Return -1 (not found)"
        },
        {},
        {"lo <= hi at every iteration", "arr[lo..hi] always contains target if present"},
        {{"time", "O(log n)"}, {"space", "O(1)"}}
    };

    reg["merge_sort"] = {
        "merge_sort",
        "Sort array by recursively splitting and merging halves",
        {{"arr", "vector<int>"}},
        {"sorted_arr", "vector<int>"},
        {
            "Base case: if len(arr) <= 1 return arr",
            "Split arr at mid into left and right halves",
            "Recursively sort left and right",
            "Merge: compare heads of left and right, take smaller, repeat until both empty"
        },
        {},
        {"left and right are both sorted before merging"},
        {{"time", "O(n log n)"}, {"space", "O(n)"}}
    };

    reg["dijkstra"] = {
        "dijkstra",
        "Single-source shortest paths in a weighted graph",
        {{"graph", "map<int,vector<pair<int,int>>>"}, {"src", "int"}},
        {"dist", "map<int,double>"},
        {
            "Initialize dist[src]=0, all others=infinity",
            "Push (0, src) onto a min-heap",
            "While heap non-empty: pop (cost, node)",
            "Skip if cost > dist[node]",
            "For each (neighbor, weight): if cost+weight < dist[neighbor]: update and push"
        },
        {"min-heap (priority_queue)", "visited set"},
        {"dist[node] always the shortest known distance"},
        {{"time", "O((V+E) log V)"}, {"space", "O(V+E)"}}
    };

    return reg;
}

inline const std::map<std::string, LogicPlan>& plan_registry() {
    static auto reg = build_plan_registry();
    return reg;
}

// ── LLMTranscriber stub: generates a prompt; actual LLM call via adapter ─────
class LLMTranscriber {
public:
    // Returns the structured prompt to send to an LLM. Actual LLM dispatch
    // happens via brain3::adapters::LLMAdapter (Java/JNI layer).
    std::string make_prompt(const LogicPlan& plan, const std::string& lang = "cpp") const {
        return plan.to_prompt(lang);
    }

    // Extract function body from LLM raw response
    std::string extract_function(const std::string& raw, const std::string& fn_name) const {
        // Try code fence first
        auto fence_start = raw.find("```");
        if (fence_start != std::string::npos) {
            auto code_start = raw.find('\n', fence_start);
            auto code_end = raw.find("```", code_start + 1);
            if (code_start != std::string::npos && code_end != std::string::npos)
                return raw.substr(code_start + 1, code_end - code_start - 1);
        }
        // Try to find function signature
        std::string sig = fn_name + "(";
        auto pos = raw.find(sig);
        if (pos != std::string::npos) return raw.substr(pos);
        return raw;
    }
};

}}}
