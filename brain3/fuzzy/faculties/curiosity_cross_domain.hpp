#pragma once
#include <string>
#include <vector>

namespace brain2 {
namespace faculties {

class CuriosityCrossDomain {
public:
    std::string generate_hypothesis(const std::string& domain_a, const std::string& domain_b) {
        return "Hypothesis: " + domain_a + " shares structural properties with " + domain_b;
    }
};

} // namespace faculties
} // namespace brain2
