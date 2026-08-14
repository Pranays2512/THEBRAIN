#pragma once
#include <vector>
#include <cmath>
#include <string>
#include <map>
#include <iostream>
#include <algorithm>

namespace brain3 {
namespace engines {
namespace synthesis {

class IrregularityDetector {
private:
    bool functional(const std::vector<std::pair<double, double>>& pairs, double tol = 1e-6) const {
        std::map<double, double> seen;
        for (const auto& [x, y] : pairs) {
            auto it = seen.find(x);
            if (it != seen.end() && std::abs(it->second - y) > tol + 1e-6 * std::abs(y))
                return false;
            seen[x] = y;
        }
        return true;
    }

    double law_error(const std::vector<std::pair<double, double>>& train,
                     const std::vector<std::pair<double, double>>& holdout) const {
        int n = train.size();
        if (n == 0) return 1e9;
        
        double sx = 0, sy = 0;
        for (const auto& [x, y] : train) { sx += x; sy += y; }
        double mx = sx / n, my = sy / n;

        double num = 0, den = 0;
        for (const auto& [x, y] : train) {
            num += (x - mx) * (y - my);
            den += (x - mx) * (x - mx);
        }
        if (den == 0) den = 1e-9;
        double m = num / den;
        double b = my - m * mx;

        double best = 1e9;
        double err_lin = 0;
        for (const auto& [x, y] : holdout) {
            double e = std::abs((m * x + b) - y) / (std::abs(y) + 1e-9);
            err_lin = std::max(err_lin, e);
        }
        best = std::min(best, err_lin);

        bool all_pos = true;
        for (const auto& [x, y] : train) if (x <= 0 || y <= 0) all_pos = false;
        for (const auto& [x, y] : holdout) if (x <= 0 || y <= 0) all_pos = false;

        if (all_pos) {
            double slx = 0, sly = 0;
            for (const auto& [x, y] : train) { slx += std::log(x); sly += std::log(y); }
            double lmx = slx / n, lmy = sly / n;

            double lnum = 0, lden = 0;
            for (const auto& [x, y] : train) {
                double lx = std::log(x), ly = std::log(y);
                lnum += (lx - lmx) * (ly - lmy);
                lden += (lx - lmx) * (lx - lmx);
            }
            if (lden == 0) lden = 1e-9;
            double p = lnum / lden;
            double k = std::exp(lmy - p * lmx);

            double err_pow = 0;
            for (const auto& [x, y] : holdout) {
                double e = std::abs(k * std::pow(x, p) - y) / (std::abs(y) + 1e-9);
                err_pow = std::max(err_pow, e);
            }
            best = std::min(best, err_pow);
        }
        return best;
    }

public:
    struct Verdict {
        std::string verdict;
        std::string why;
    };

    Verdict assess(const std::vector<std::pair<double, double>>& train,
                   const std::vector<std::pair<double, double>>& holdout,
                   double law_tol = 0.05) const {
        if (train.empty() || holdout.empty())
            return {"IRREGULAR", "insufficient data (need train + held-out points)"};
        
        std::vector<std::pair<double, double>> all = train;
        all.insert(all.end(), holdout.begin(), holdout.end());
        if (!functional(all))
            return {"IRREGULAR", "not even a function (same input -> different outputs)"};
            
        double err = law_error(train, holdout);
        if (err <= law_tol) {
            char buf[128];
            snprintf(buf, sizeof(buf), "a law predicts held-out within %.0f%% (err %.1f%%)", law_tol * 100, err * 100);
            return {"REGULAR", buf};
        }
        
        char buf[128];
        snprintf(buf, sizeof(buf), "no law predicts held-out (best err %.0f%%) -> unverifiable", err * 100);
        return {"IRREGULAR", buf};
    }
};

}}}
