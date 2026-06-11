#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace brain2 {

struct SparseElement {
    int index;
    float value;
};

class SparseVector {
public:
    std::vector<SparseElement> elements;
    int size_; // Logical dense size

    SparseVector() : size_(0) {}
    SparseVector(int size) : size_(size) {}

    // Compress from dense
    static SparseVector from_dense(const std::vector<float>& dense, float threshold = 1e-6f) {
        SparseVector sv(dense.size());
        for (int i = 0; i < (int)dense.size(); i++) {
            if (std::abs(dense[i]) > threshold) {
                sv.elements.push_back({i, dense[i]});
            }
        }
        return sv;
    }

    // Decompress to dense
    std::vector<float> to_dense() const {
        std::vector<float> dense(size_, 0.f);
        for (const auto& el : elements) {
            if (el.index >= 0 && el.index < size_) {
                dense[el.index] = el.value;
            }
        }
        return dense;
    }

    // Fast dot product with a dense vector
    float dot_dense(const std::vector<float>& dense) const {
        float sum = 0.f;
        for (const auto& el : elements) {
            if (el.index < (int)dense.size()) {
                sum += el.value * dense[el.index];
            }
        }
        return sum;
    }

    // Fast sparse-sparse dot product (assumes sorted indices)
    float dot_sparse(const SparseVector& other) const {
        float sum = 0.f;
        int i = 0, j = 0;
        int n1 = elements.size(), n2 = other.elements.size();
        while (i < n1 && j < n2) {
            if (elements[i].index == other.elements[j].index) {
                sum += elements[i].value * other.elements[j].value;
                i++;
                j++;
            } else if (elements[i].index < other.elements[j].index) {
                i++;
            } else {
                j++;
            }
        }
        return sum;
    }

    void clear() {
        elements.clear();
    }
    
    bool empty() const {
        return elements.empty();
    }
};

} // namespace brain2
