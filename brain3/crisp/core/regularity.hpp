#pragma once
// regularity.hpp — native port of irregularity_detector._law_error. Fits a linear and a
// power law to training (x,y) points by least squares and returns the best held-out relative
// error. This is the math behind "does this domain have a checkable regularity" — the brain's
// map of where verification can reach. Pure numeric, deterministic, proven in Python.
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace brain2 {

// Best held-out max-relative-error fitting y=m*x+b and y=k*x^p (whichever is lower).
inline double law_error(const std::vector<std::pair<double, double>>& train,
                        const std::vector<std::pair<double, double>>& hold) {
    int n = (int)train.size();
    if (n == 0 || hold.empty()) return 1e9;
    double best = 1e9;

    // linear least squares
    double mx = 0, my = 0;
    for (auto& p : train) { mx += p.first; my += p.second; }
    mx /= n; my /= n;
    double den = 0, num = 0;
    for (auto& p : train) { den += (p.first - mx) * (p.first - mx); num += (p.first - mx) * (p.second - my); }
    if (den == 0) den = 1e-9;
    double m = num / den, b = my - m * mx;
    double err = 0;
    for (auto& p : hold) err = std::max(err, std::fabs((m * p.first + b) - p.second) / (std::fabs(p.second) + 1e-9));
    best = std::min(best, err);

    // power law via log-log (positives only)
    bool pos = true;
    for (auto& p : train) if (p.first <= 0 || p.second <= 0) pos = false;
    for (auto& p : hold)  if (p.first <= 0 || p.second <= 0) pos = false;
    if (pos) {
        double lmx = 0, lmy = 0;
        for (auto& p : train) { lmx += std::log(p.first); lmy += std::log(p.second); }
        lmx /= n; lmy /= n;
        double lden = 0, lnum = 0;
        for (auto& p : train) {
            double lx = std::log(p.first);
            lden += (lx - lmx) * (lx - lmx);
            lnum += (lx - lmx) * (std::log(p.second) - lmy);
        }
        if (lden == 0) lden = 1e-9;
        double pp = lnum / lden, k = std::exp(lmy - pp * lmx);
        double e2 = 0;
        for (auto& p : hold) e2 = std::max(e2, std::fabs(k * std::pow(p.first, pp) - p.second) / (std::fabs(p.second) + 1e-9));
        best = std::min(best, e2);
    }
    return best;
}

}  // namespace brain2
