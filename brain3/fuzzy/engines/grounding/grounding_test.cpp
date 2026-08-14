#include <iostream>
#include <iomanip>
#include "grounding.hpp"
#include "context_embed.hpp"
#include "ground_blend.hpp"

using namespace brain2::grounding;

void test_grounding() {
    std::cout << "=== grounding — Numeric Calibration ===\n\n";
    GroundingPipeline pipe(32);
    
    std::vector<std::string> attrs = {"mass", "speed"};
    std::vector<std::map<std::string, float>> obs = {
        {{"mass", 10.0f}, {"speed", 5.0f}},
        {{"mass", 20.0f}, {"speed", 2.0f}},
        {{"mass", 5.0f},  {"speed", 10.0f}},
        {{"mass", 15.0f}, {"speed", 7.0f}}
    };
    
    pipe.calibrate_numeric_sensors(attrs, obs);
    
    // Test decode on an observation's vector
    // Reconstruct V manually for the first observation to decode
    Vector v(32, 0.0f);
    for (int i=0; i<32; i++) {
        v[i] += 10.0f * pipe.numeric_axes["mass"][i] + 5.0f * pipe.numeric_axes["speed"][i];
    }
    
    auto dec = pipe.decode_numeric(v, attrs);
    std::cout << "Decoded mass (expected ~10.0): " << dec["mass"] << "\n";
    std::cout << "Decoded speed (expected ~5.0): " << dec["speed"] << "\n";
}

void test_context_embed() {
    std::cout << "\n=== context_embed — meaning from context ===\n\n";
    std::vector<std::string> corpus = {
        "the rocket has high speed",
        "the rocket travels at high velocity",
        "the engine makes strong force",
        "the engine makes strong push",
        "a strong push is a large force"
    };
    
    auto vecs = build_context_embeddings(corpus);
    std::vector<std::string> canonicals = {"speed", "force"};
    
    for (std::string w : {"velocity", "push"}) {
        auto res = nearest_canonical(w, canonicals, vecs);
        std::cout << "  '" << w << "' --context--> '" << res.first << "' (sim: " << res.second << ")\n";
    }
}

int main() {
    test_grounding();
    test_context_embed();
    return 0;
}
