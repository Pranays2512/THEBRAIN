#pragma once
#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <iostream>

namespace brain2 {
namespace math {

using DimVec = std::vector<int>;

inline DimVec d_add(const DimVec& a, const DimVec& b) {
    DimVec r(a.size());
    for(size_t i=0; i<a.size(); ++i) r[i] = a[i] + b[i];
    return r;
}

inline DimVec d_sub(const DimVec& a, const DimVec& b) {
    DimVec r(a.size());
    for(size_t i=0; i<a.size(); ++i) r[i] = a[i] - b[i];
    return r;
}

inline DimVec d_scale(const DimVec& a, int n) {
    DimVec r(a.size());
    for(size_t i=0; i<a.size(); ++i) r[i] = a[i] * n;
    return r;
}

inline DimVec get_units(const std::string& name, const std::map<std::string, DimVec>& units) {
    if (units.count(name)) return units.at(name);
    throw std::runtime_error("unknown unit: " + name);
}

// In the actual engine, this would recursively traverse ExprNode and do the dimension math.
// Here we just provide a simplified string-based helper to test the concept.
inline DimVec parse_units(const std::string& op, const DimVec& a, const DimVec& b = {}) {
    if (op == "*") return d_add(a, b);
    if (op == "/") return d_sub(a, b);
    if (op == "+" || op == "-") {
        if (a != b) throw std::runtime_error("dimensional mismatch");
        return a;
    }
    throw std::runtime_error("unknown op for units");
}

} // namespace math
} // namespace brain2
