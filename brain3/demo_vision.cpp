#include <iostream>
#include <fstream>
#include <cmath>
#include "crisp/engines/vision/vision_engine.hpp"
#include "crisp/engines/reasoning/reasoning_engine.hpp"

using namespace brain3::engines::vision;
using namespace brain2::reasoning;

void generate_test_image(const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    int w = 300, h = 300;
    // P6 = binary RGB, maxval 255
    f << "P6\n" << w << " " << h << "\n255\n";
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            // Draw a red square on the left
            if (x >= 30 && x <= 100 && y >= 100 && y <= 170) {
                f << (char)255 << (char)0 << (char)0;
            } 
            // Draw a blue circle on the right
            else if (std::pow(x - 220, 2) + std::pow(y - 150, 2) <= 1600) {
                f << (char)0 << (char)0 << (char)255;
            } 
            // Draw a small green dot in the center
            else if (std::pow(x - 150, 2) + std::pow(y - 50, 2) <= 100) {
                f << (char)0 << (char)255 << (char)0;
            }
            // Background white
            else {
                f << (char)255 << (char)255 << (char)255;
            }
        }
    }
}

int main() {
    std::cout << "1. Generating test image (test_vision.ppm)...\n";
    std::string img_path = "test_vision.ppm";
    generate_test_image(img_path);
    
    std::cout << "2. Initializing Brain Reasoning Engine...\n";
    ReasoningEngine kb;
    VisionEngine vision;

    std::cout << "3. Parsing image and injecting facts...\n";
    auto blobs = vision.parse_image(img_path, kb);

    std::cout << "\nExtracted Physical Blobs:\n";
    for (const auto& b : blobs) {
        std::cout << "  Blob " << b.id << ": color=" << b.color_name 
                  << ", pos=" << b.position_name 
                  << ", area=" << b.area << "\n";
    }

    std::cout << "\n4. Querying Brain Reasoning Engine...\n";
    
    // Let's ask some logical queries based on what it just saw.
    auto ask = [&](const std::string& subj, const std::string& rel) {
        auto [ans, reason] = kb.ask(subj, rel);
        if (ans.empty()) {
            std::cout << "  ? " << subj << "." << rel << " -> (I don't know)\n";
        } else {
            std::cout << "  > " << subj << "." << rel << " = " << ans << "\n";
        }
    };

    ask("blob_1", "color");
    ask("blob_1", "position");
    ask("blob_2", "color");
    ask("blob_2", "size");
    ask("blob_3", "position");

    std::cout << "\nQuerying backward: Which object is blue?\n";
    bool found_blue = false;
    for (const auto& f : kb.facts) {
        if (f.rel == "color" && f.obj == "blue") {
            std::cout << "  > " << f.subj << " is blue.\n";
            found_blue = true;
        }
    }
    if (!found_blue) std::cout << "  > No blue object seen.\n";

    return 0;
}
