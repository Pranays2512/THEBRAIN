#pragma once

#include <string>
#include <vector>
#include <map>
#include <queue>
#include <iostream>
#include <memory>
#include <cmath>

#ifndef STB_IMAGE_STATIC
#define STB_IMAGE_STATIC
#endif
#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif
#include "../../../vendor/stb_image.h"
#include "../../engines/reasoning/reasoning_engine.hpp"

namespace brain3 {
namespace engines {
namespace vision {

struct Blob {
    int id;
    int min_x, min_y, max_x, max_y;
    int area;
    int r, g, b; 
    std::string color_name;
    std::string position_name;
};

class VisionEngine {
private:
    std::string classify_color(int r, int g, int b) {
        if (r > 150 && g < 100 && b < 100) return "red";
        if (g > 150 && r < 100 && b < 100) return "green";
        if (b > 150 && r < 100 && g < 100) return "blue";
        if (r > 200 && g > 200 && b > 200) return "white";
        if (r < 50 && g < 50 && b < 50) return "black";
        if (r > 150 && g > 150 && b < 100) return "yellow";
        return "unknown_color";
    }

    std::string classify_position(int center_x, int img_width) {
        if (center_x < img_width / 3) return "left";
        if (center_x > 2 * img_width / 3) return "right";
        return "center";
    }

    std::string classify_size(int area, int total_area) {
        double ratio = (double)area / total_area;
        if (ratio > 0.05) return "large";
        if (ratio > 0.01) return "medium";
        return "small";
    }

public:
    std::vector<Blob> parse_image(const std::string& path, brain2::reasoning::ReasoningEngine& kb) {
        int width, height, channels;
        unsigned char* img = stbi_load(path.c_str(), &width, &height, &channels, 3);
        if (!img) {
            std::cerr << "VisionEngine Error: Failed to load image " << path << "\n";
            return {};
        }

        std::vector<Blob> blobs;
        std::vector<bool> visited(width * height, false);

        auto get_pixel = [&](int x, int y) {
            int idx = (y * width + x) * 3;
            return std::make_tuple(img[idx], img[idx+1], img[idx+2]);
        };

        auto is_similar = [](const std::tuple<int,int,int>& p1, const std::tuple<int,int,int>& p2) {
            return std::abs(std::get<0>(p1) - std::get<0>(p2)) < 30 &&
                   std::abs(std::get<1>(p1) - std::get<1>(p2)) < 30 &&
                   std::abs(std::get<2>(p1) - std::get<2>(p2)) < 30;
        };

        int blob_count = 0;

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (visited[y * width + x]) continue;

                auto base_color = get_pixel(x, y);
                auto color_name = classify_color(std::get<0>(base_color), std::get<1>(base_color), std::get<2>(base_color));
                
                // Ignore background colors
                if (color_name == "white" || color_name == "black") {
                    visited[y * width + x] = true;
                    continue; 
                }

                Blob b;
                b.id = ++blob_count;
                b.min_x = x; b.max_x = x;
                b.min_y = y; b.max_y = y;
                b.area = 0;
                long long sum_r = 0, sum_g = 0, sum_b = 0;

                std::queue<std::pair<int, int>> q;
                q.push({x, y});
                visited[y * width + x] = true;

                while (!q.empty()) {
                    auto [cx, cy] = q.front();
                    q.pop();

                    b.min_x = std::min(b.min_x, cx);
                    b.max_x = std::max(b.max_x, cx);
                    b.min_y = std::min(b.min_y, cy);
                    b.max_y = std::max(b.max_y, cy);
                    b.area++;

                    auto c_color = get_pixel(cx, cy);
                    sum_r += std::get<0>(c_color);
                    sum_g += std::get<1>(c_color);
                    sum_b += std::get<2>(c_color);

                    int dx[] = {-1, 1, 0, 0};
                    int dy[] = {0, 0, -1, 1};
                    for (int i = 0; i < 4; ++i) {
                        int nx = cx + dx[i];
                        int ny = cy + dy[i];
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            if (!visited[ny * width + nx]) {
                                auto n_color = get_pixel(nx, ny);
                                if (is_similar(base_color, n_color)) {
                                    visited[ny * width + nx] = true;
                                    q.push({nx, ny});
                                }
                            }
                        }
                    }
                }

                // Filter out tiny artifacts
                if (b.area > 50) { 
                    b.r = sum_r / b.area;
                    b.g = sum_g / b.area;
                    b.b = sum_b / b.area;
                    b.color_name = classify_color(b.r, b.g, b.b);
                    b.position_name = classify_position((b.min_x + b.max_x) / 2, width);
                    blobs.push_back(b);
                }
            }
        }

        stbi_image_free(img);

        // Grounding phase: Translate pixel data into symbolic facts in the knowledge base
        int total_area = width * height;
        for (const auto& b : blobs) {
            std::string obj = "blob_" + std::to_string(b.id);
            kb.learn(obj, "is_a", "object");
            if (b.color_name != "unknown_color") {
                kb.learn(obj, "color", b.color_name);
            }
            kb.learn(obj, "position", b.position_name);
            kb.learn(obj, "size", classify_size(b.area, total_area));
        }

        return blobs;
    }
};

}
}
}
