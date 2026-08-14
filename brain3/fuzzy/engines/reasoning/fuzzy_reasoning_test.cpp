#include <iostream>
#include <iomanip>
#include "tree_learn.hpp"

using namespace brain2::reasoning;

int main() {
    std::cout << "=== tree_learn — heuristic weight learning ===\n\n";
    
    // Simulate learning heuristic weights from state features (like Manhattan distance components)
    TreeLearn learner;
    
    // Fake training data: 3 features, 4 samples. 
    // True cost y = 2*f0 + 1.5*f1 + 0.5*f2
    std::vector<std::vector<double>> X = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
        {1.0, 1.0, 1.0}
    };
    std::vector<double> y = {2.0, 1.5, 0.5, 4.0};
    
    learner.fit(X, y);
    
    std::cout << "Learned weights (should be approx 2.0, 1.5, 0.5):\n";
    for (size_t i = 0; i < learner.weights.size(); ++i) {
        std::cout << "  w[" << i << "] = " << std::setprecision(3) << learner.weights[i] << "\n";
    }
    
    std::cout << "\nTest Prediction on {2, 2, 2}:\n";
    std::cout << "  Expected: ~8.0\n";
    std::cout << "  Predicted: " << learner.predict({2.0, 2.0, 2.0}) << "\n";
    
    return 0;
}
