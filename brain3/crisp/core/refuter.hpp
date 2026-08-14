#pragma once
// refuter.hpp — native port of the refuter's core (refuter.py): given a candidate's outputs
// and the oracle's outputs over a contiguous integer input range, find WHERE the candidate
// breaks and characterise the VALID SCOPE (the integer intervals where it holds). The
// verifier turned aggressive — hunts disagreement, not agreement. Hot path; proven in Python.
#include <climits>
#include <string>
#include <utility>
#include <vector>

namespace brain2 {

struct RefuteResult {
    bool        robust = true;      // no disagreement found
    long        breaks_at = LONG_MIN;  // first input where candidate != oracle (LONG_MIN = none)
    double      fail_rate = 0.0;    // fraction of inputs that disagreed
    std::string scope;              // human description of the valid range
};

// cand[i], oracle[i] are the two functions' outputs for input lo+i.
inline RefuteResult refute_int1(const std::vector<long>& cand,
                                const std::vector<long>& oracle, long lo) {
    RefuteResult r;
    long n = (long)std::min(cand.size(), oracle.size());
    long hi = lo + n - 1;
    std::vector<std::pair<long, long>> runs;
    long run_start = 0, prev = 0;
    bool inrun = false;
    long fails = 0;
    for (long i = 0; i < n; i++) {
        long x = lo + i;
        if (cand[i] == oracle[i]) {
            if (!inrun) { run_start = x; inrun = true; }
            prev = x;
        } else {
            fails++;
            if (r.breaks_at == LONG_MIN) r.breaks_at = x;
            if (inrun) { runs.push_back({run_start, prev}); inrun = false; }
        }
    }
    if (inrun) runs.push_back({run_start, prev});
    r.robust = (r.breaks_at == LONG_MIN);
    r.fail_rate = n ? (double)fails / (double)n : 0.0;

    // scope string — matches refuter.py's _int1_scope formatting
    if (runs.empty()) {
        r.scope = "holds nowhere in [" + std::to_string(lo) + "," + std::to_string(hi) + "]";
    } else if (runs.size() == 1 && runs[0].first == lo && runs[0].second == hi) {
        r.scope = "holds for ALL n in [" + std::to_string(lo) + "," + std::to_string(hi) + "]";
    } else {
        std::string s;
        for (size_t k = 0; k < runs.size(); k++) {
            if (k) s += " ∪ ";   // ' ∪ '
            long a = runs[k].first, b = runs[k].second;
            s += (a == b) ? ("{" + std::to_string(a) + "}")
                          : ("[" + std::to_string(a) + "," + std::to_string(b) + "]");
        }
        r.scope = "holds for n in " + s;
    }
    return r;
}

}  // namespace brain2
