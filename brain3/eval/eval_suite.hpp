#pragma once
/**
 * brain3/eval/eval_suite.hpp — measurement infrastructure for the architecture.
 *
 * Every capability claim must be graded here or it does not exist. Suites are
 * plain functions returning SuiteResult; run_eval aggregates them into a JSON
 * scorecard and enforces regression gates (exit code).
 */
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <iomanip>

namespace brain3 {
namespace eval {

struct SuiteResult {
    std::string name;
    double score = 0.0;              // headline metric in [0,1]
    bool   gate_pass = true;
    std::vector<std::pair<std::string, std::string>> detail;  // key,value rows
};

inline std::string json_escape(const std::string& s) {
    std::ostringstream o;
    for (char c : s) {
        switch (c) {
            case '"': o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    o << buf;
                } else o << c;
        }
    }
    return o.str();
}

inline std::string render_report(const std::vector<SuiteResult>& suites,
                                 bool gates_pass, double wall_seconds) {
    std::ostringstream o;
    o << "{\n  \"gates_pass\": " << (gates_pass ? "true" : "false") << ",\n";
    o << "  \"wall_seconds\": " << std::fixed << std::setprecision(1)
      << wall_seconds << ",\n  \"suites\": {\n";
    for (size_t i = 0; i < suites.size(); ++i) {
        const auto& s = suites[i];
        o << "    \"" << json_escape(s.name) << "\": {\n";
        o << "      \"score\": " << std::fixed << std::setprecision(4)
          << s.score << ",\n";
        o << "      \"gate_pass\": " << (s.gate_pass ? "true" : "false");
        if (!s.detail.empty()) o << ",\n      \"detail\": {";
        for (size_t j = 0; j < s.detail.size(); ++j) {
            o << (j ? ", " : "\n        ")
              << '"' << json_escape(s.detail[j].first) << "\": \""
              << json_escape(s.detail[j].second) << '"';
            if (j + 1 == s.detail.size()) o << "\n      }";
        }
        o << "\n    }" << (i + 1 < suites.size() ? "," : "") << "\n";
    }
    o << "  }\n}\n";
    return o.str();
}

// first token of a BQL string = its op family
inline std::string op_family(const std::string& bql) {
    size_t sp = bql.find(' ');
    return sp == std::string::npos ? bql : bql.substr(0, sp);
}

} // namespace eval
} // namespace brain3
