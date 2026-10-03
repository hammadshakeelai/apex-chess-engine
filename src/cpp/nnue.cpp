#include "nnue.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace Apex {

NNUEModel GlobalNNUE;

NNUEModel::NNUEModel()
    : loaded(false), feature_dim(40960), hidden_dim(256),
      feature_weights(nullptr), feature_bias(nullptr),
      fc1_weights(nullptr), fc1_bias(nullptr),
      fc2_weights(nullptr), fc2_bias(nullptr),
      out_weights(nullptr), out_bias(0.0f) {}

NNUEModel::~NNUEModel() {
    delete[] feature_weights;
    delete[] feature_bias;
    delete[] fc1_weights;
    delete[] fc1_bias;
    delete[] fc2_weights;
    delete[] fc2_bias;
    delete[] out_weights;
}

bool NNUEModel::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        return false;
    }

    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, "APEXNNUE", 8) != 0) {
        return false;
    }

    uint32_t version;
    f.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    f.read(reinterpret_cast<char*>(&feature_dim), sizeof(uint32_t));
    f.read(reinterpret_cast<char*>(&hidden_dim), sizeof(uint32_t));

    // Allocate buffers
    delete[] feature_weights;
    delete[] feature_bias;
    delete[] fc1_weights;
    delete[] fc1_bias;
    delete[] fc2_weights;
    delete[] fc2_bias;
    delete[] out_weights;

    feature_weights = new int16_t[feature_dim * hidden_dim];
    feature_bias = new int16_t[hidden_dim];
    fc1_weights = new float[512 * 32];
    fc1_bias = new float[32];
    fc2_weights = new float[32 * 32];
    fc2_bias = new float[32];
    out_weights = new float[32];

    f.read(reinterpret_cast<char*>(feature_weights), feature_dim * hidden_dim * sizeof(int16_t));
    f.read(reinterpret_cast<char*>(feature_bias), hidden_dim * sizeof(int16_t));
    f.read(reinterpret_cast<char*>(fc1_weights), 512 * 32 * sizeof(float));
    f.read(reinterpret_cast<char*>(fc1_bias), 32 * sizeof(float));
    f.read(reinterpret_cast<char*>(fc2_weights), 32 * 32 * sizeof(float));
    f.read(reinterpret_cast<char*>(fc2_bias), 32 * sizeof(float));
    f.read(reinterpret_cast<char*>(out_weights), 32 * sizeof(float));
    f.read(reinterpret_cast<char*>(&out_bias), sizeof(float));

    if (!f.good()) {
        loaded = false;
        return false;
    }

    loaded = true;
    return true;
}

static inline int halfkp_pt(Piece p, bool is_white_perspective) {
    PieceType pt = piece_type(p);
    if (pt == KING) return -1;

    int type_offset = 0;
    switch (pt) {
        case PAWN:   type_offset = 0; break;
        case KNIGHT: type_offset = 1; break;
        case BISHOP: type_offset = 2; break;
        case ROOK:   type_offset = 3; break;
        case QUEEN:  type_offset = 4; break;
        default: return -1;
    }

    bool is_friendly = (piece_color(p) == WHITE) ? is_white_perspective : !is_white_perspective;
    return is_friendly ? type_offset : (type_offset + 5);
}

void NNUEModel::extract_indices(const Position& pos, std::vector<int>& w_idx, std::vector<int>& b_idx) const {
    Square w_ksq = pos.king_square(WHITE);
    Square b_ksq = pos.king_square(BLACK);

    int b_ksq_mirrored = b_ksq ^ 56;

    for (int sq = 0; sq < 64; ++sq) {
        Piece p = pos.get_piece_at(static_cast<Square>(sq));
        if (p == NO_PIECE) continue;

        int w_pt = halfkp_pt(p, true);
        if (w_pt >= 0) {
            int idx = w_ksq * 640 + (w_pt * 64 + sq);
            w_idx.push_back(idx);
        }

        int b_pt = halfkp_pt(p, false);
        if (b_pt >= 0) {
            int sq_mirrored = sq ^ 56;
            int idx = b_ksq_mirrored * 640 + (b_pt * 64 + sq_mirrored);
            b_idx.push_back(idx);
        }
    }
}

static inline float screlu(float x) {
    float c = std::clamp(x, 0.0f, 1.0f);
    return c * c;
}

int NNUEModel::evaluate(const Position& pos) const {
    if (!loaded) return 0;

    alignas(32) int16_t acc_w[256];
    alignas(32) int16_t acc_b[256];

    std::memcpy(acc_w, feature_bias, 256 * sizeof(int16_t));
    std::memcpy(acc_b, feature_bias, 256 * sizeof(int16_t));

    std::vector<int> w_idx;
    std::vector<int> b_idx;
    w_idx.reserve(32);
    b_idx.reserve(32);
    extract_indices(pos, w_idx, b_idx);

    // AVX2 SIMD Feature Transformer Accumulation
    for (int idx : w_idx) {
        const int16_t* w_row = &feature_weights[idx * 256];
        for (int i = 0; i < 256; i += 16) {
            __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(&acc_w[i]));
            __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&w_row[i]));
            a = _mm256_add_epi16(a, w);
            _mm256_store_si256(reinterpret_cast<__m256i*>(&acc_w[i]), a);
        }
    }

    for (int idx : b_idx) {
        const int16_t* b_row = &feature_weights[idx * 256];
        for (int i = 0; i < 256; i += 16) {
            __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(&acc_b[i]));
            __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&b_row[i]));
            a = _mm256_add_epi16(a, w);
            _mm256_store_si256(reinterpret_cast<__m256i*>(&acc_b[i]), a);
        }
    }

    // Activations SCReLU (float normalized by 255.0)
    float act_w[256];
    float act_b[256];
    for (int i = 0; i < 256; ++i) {
        act_w[i] = screlu(acc_w[i] / 255.0f);
        act_b[i] = screlu(acc_b[i] / 255.0f);
    }

    // Stack side to move first
    float hidden[512];
    if (pos.turn() == WHITE) {
        std::memcpy(&hidden[0], act_w, 256 * sizeof(float));
        std::memcpy(&hidden[256], act_b, 256 * sizeof(float));
    } else {
        std::memcpy(&hidden[0], act_b, 256 * sizeof(float));
        std::memcpy(&hidden[256], act_w, 256 * sizeof(float));
    }

    // Dense Layer 1: hidden [512] x fc1_weights [512, 32] + fc1_bias [32]
    float h1[32];
    for (int j = 0; j < 32; ++j) {
        float sum = fc1_bias[j];
        for (int i = 0; i < 512; ++i) {
            sum += hidden[i] * fc1_weights[i * 32 + j];
        }
        h1[j] = screlu(sum);
    }

    // Dense Layer 2: h1 [32] x fc2_weights [32, 32] + fc2_bias [32]
    float h2[32];
    for (int j = 0; j < 32; ++j) {
        float sum = fc2_bias[j];
        for (int i = 0; i < 32; ++i) {
            sum += h1[i] * fc2_weights[i * 32 + j];
        }
        h2[j] = screlu(sum);
    }

    // Output Centipawn Score (scaled by 400.0)
    float out = out_bias;
    for (int i = 0; i < 32; ++i) {
        out += h2[i] * out_weights[i];
    }

    return static_cast<int>(std::round(out * 400.0f));
}

} // namespace Apex
