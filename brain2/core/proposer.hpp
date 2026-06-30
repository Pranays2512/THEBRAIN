#pragma once
// proposer.hpp — native port of the learned proposer's scoring (feature_learner.py):
// disc_weights (a feature's weight = its variance across the winning-space profiles, i.e.
// how well it SEPARATES spaces) and feat_sim (weighted similarity of a task's features to a
// space profile). This is the hot scoring the proposer runs to order synthesis spaces.
#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace brain2 {

// Each feature's weight = variance across the space profiles (discriminative power),
// normalised. Empty if fewer than 2 profiles (caller falls back to uniform).
inline std::map<std::string, double> disc_weights(
        const std::vector<std::map<std::string, double>>& protos) {
    std::map<std::string, double> w;
    if (protos.size() < 2) return w;
    double tot = 0.0;
    for (const auto& kv : protos[0]) {
        const std::string& fk = kv.first;
        double mu = 0.0;
        for (const auto& p : protos) { auto it = p.find(fk); mu += (it != p.end() ? it->second : 0.0); }
        mu /= (double)protos.size();
        double var = 0.0;
        for (const auto& p : protos) {
            auto it = p.find(fk);
            double v = (it != p.end() ? it->second : 0.0);
            var += (v - mu) * (v - mu);
        }
        var /= (double)protos.size();
        w[fk] = var;
        tot += var;
    }
    if (tot <= 0.0) tot = 1.0;
    for (auto& kv : w) kv.second /= tot;
    return w;
}

// Weighted similarity of a task's features to a space profile. If w is empty -> uniform.
inline double feat_sim(const std::map<std::string, double>& feats,
                       const std::map<std::string, double>& proto,
                       const std::map<std::string, double>& w) {
    double num = 0.0, den = 0.0;
    for (const auto& kv : feats) {
        const std::string& fk = kv.first;
        double fv = kv.second;
        double wt = 1.0;
        if (!w.empty()) { auto it = w.find(fk); wt = (it != w.end() ? it->second : 0.0); }
        auto pit = proto.find(fk);
        double pv = (pit != proto.end() ? pit->second : 0.0);
        num += wt * (1.0 - std::fabs(fv - pv));
        den += wt;
    }
    return num / (den != 0.0 ? den : 1.0);
}

}  // namespace brain2
