#pragma once

#include "types.h"
#include "position.h"
#include <string>
#include <vector>
#include <immintrin.h>

namespace Apex {

class NNUEModel {
public:
    NNUEModel();
    ~NNUEModel();

    bool load(const std::string& path);
    bool is_loaded() const { return loaded; }

    int evaluate(const Position& pos) const;

private:
    bool loaded;
    uint32_t feature_dim;
    uint32_t hidden_dim;

    // Feature Transformer
    int16_t* feature_weights;  // [40960, 256]
    int16_t* feature_bias;     // [256]

    // Dense Layer 1: [512, 32]
    float* fc1_weights;
    float* fc1_bias;

    // Dense Layer 2: [32, 32]
    float* fc2_weights;
    float* fc2_bias;

    // Output Layer: [32]
    float* out_weights;
    float out_bias;

    void extract_indices(const Position& pos, std::vector<int>& w_idx, std::vector<int>& b_idx) const;
};

extern NNUEModel GlobalNNUE;

} // namespace Apex
