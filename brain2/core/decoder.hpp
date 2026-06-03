#pragma once

#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <stdexcept>
#include <random>
#include "language.hpp"

namespace brain2 {

// A lightweight Recurrent Neural Network (Elman RNN) for Sequence-to-Sequence generation
struct DecoderRNN {
    int input_size;
    int hidden_size;
    int output_size;
    float lr_;

    std::vector<float> W_ih;
    std::vector<float> W_hh;
    std::vector<float> b_h;
    
    std::vector<float> W_ho;
    std::vector<float> b_o;
    
    std::mt19937 rng_;

    DecoderRNN() : rng_(42) {}
    DecoderRNN(int in, int hid, int out, float lr=0.01f) 
        : input_size(in), hidden_size(hid), output_size(out), lr_(lr), rng_(42) {
        
        std::normal_distribution<float> nd(0.f, 0.05f);
        
        W_ih.resize(hidden_size * input_size); for(auto& w : W_ih) w = nd(rng_);
        W_hh.resize(hidden_size * hidden_size); for(auto& w : W_hh) w = nd(rng_);
        b_h.resize(hidden_size, 0.f);
        
        W_ho.resize(output_size * hidden_size); for(auto& w : W_ho) w = nd(rng_);
        b_o.resize(output_size, 0.f);
    }

    // Single step forward
    std::vector<float> step(const std::vector<float>& x_t, std::vector<float>& h_prev) const {
        std::vector<float> h_next(hidden_size, 0.f);
        
        for (int i = 0; i < hidden_size; i++) {
            float sum = b_h[i];
            for (int j = 0; j < input_size; j++) {
                sum += W_ih[i * input_size + j] * x_t[j];
            }
            for (int j = 0; j < hidden_size; j++) {
                sum += W_hh[i * hidden_size + j] * h_prev[j];
            }
            h_next[i] = std::tanh(sum);
        }
        
        h_prev = h_next;
        
        std::vector<float> y_t(output_size, 0.f);
        for (int i = 0; i < output_size; i++) {
            float sum = b_o[i];
            for (int j = 0; j < hidden_size; j++) {
                sum += W_ho[i * hidden_size + j] * h_next[j];
            }
            y_t[i] = sum;
        }
        
        return y_t;
    }
    
    // Autoregressive generation
    std::vector<std::string> generate(const std::vector<float>& initial_context, 
                                      Language& lang, 
                                      int max_len = 5) {
        std::vector<std::string> sentence;
        std::vector<float> h_prev(hidden_size, 0.f);
        std::vector<float> x_t = initial_context;
        
        for (int step = 0; step < max_len; step++) {
            std::vector<float> y_t = this->step(x_t, h_prev);
            std::string word = lang.best_word(y_t);
            if (word.empty() || word == "HALT") break;
            sentence.push_back(word);
            x_t = y_t; // feed output as next input
        }
        return sentence;
    }
    
    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) return;
        f.write((const char*)&input_size, sizeof(int));
        f.write((const char*)&hidden_size, sizeof(int));
        f.write((const char*)&output_size, sizeof(int));
        f.write((const char*)W_ih.data(), W_ih.size() * sizeof(float));
        f.write((const char*)W_hh.data(), W_hh.size() * sizeof(float));
        f.write((const char*)b_h.data(), b_h.size() * sizeof(float));
        f.write((const char*)W_ho.data(), W_ho.size() * sizeof(float));
        f.write((const char*)b_o.data(), b_o.size() * sizeof(float));
    }
    
    static DecoderRNN load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot load DecoderRNN");
        DecoderRNN d;
        f.read((char*)&d.input_size, sizeof(int));
        f.read((char*)&d.hidden_size, sizeof(int));
        f.read((char*)&d.output_size, sizeof(int));
        d.W_ih.resize(d.hidden_size * d.input_size);
        d.W_hh.resize(d.hidden_size * d.hidden_size);
        d.b_h.resize(d.hidden_size);
        d.W_ho.resize(d.output_size * d.hidden_size);
        d.b_o.resize(d.output_size);
        
        f.read((char*)d.W_ih.data(), d.W_ih.size() * sizeof(float));
        f.read((char*)d.W_hh.data(), d.W_hh.size() * sizeof(float));
        f.read((char*)d.b_h.data(), d.b_h.size() * sizeof(float));
        f.read((char*)d.W_ho.data(), d.W_ho.size() * sizeof(float));
        f.read((char*)d.b_o.data(), d.b_o.size() * sizeof(float));
        return d;
    }
};

} // namespace brain2
