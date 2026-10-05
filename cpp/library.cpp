/*
 * Copyright (c) 2026 Yuanjie Li. All rights reserved.
 * Author: Yuanjie Li
 * See LICENSE in the project root for terms.
 */

#include "library.h"
#include "9mer_levels_v1.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <eigen3/Eigen/Dense>


#include "mymath.hpp"
std::vector<float> sliding_vars(const float* data, const int data_length, int window_size) {
    if (window_size % 2 == 0) {
        window_size += 1;
    }
    if (data_length < window_size || window_size == 0) return {};
    std::vector<float> res;
    res.reserve(data_length);
    double sum = 0.0;
    double sum_sq = 0.0;
    double ws = 0.0;
    const int pad = window_size / 2;
    for (int i = 0; i < pad; ++i) {
        res.push_back(0.0);
    }
    for (size_t i = 0; i < data_length; ++i) {
        sum += data[i];
        sum_sq += data[i] * data[i];
        ws += 1.0;
        if (i >= window_size) {
            double old = data[i - window_size];
            sum -= old;
            sum_sq -= old * old;
            ws -= 1.0;
        }
        if (i >= window_size - 1) {
            const double mean = sum / ws;
            double var = (sum_sq / ws) - (mean * mean);
            res.push_back(var);
        }
    }
    const float last_val = res.back();
    for (int i = 0; i < pad; ++i) {
        res[i] = res[pad];
        res.push_back(last_val);
    }
    float min_positive = 1e5;
    for (int i = 0; i < data_length; ++i) {
        if (res[i] > 0 && res[i] < min_positive) {
            min_positive = res[i];
        }
    }
    for (int i = 0; i < data_length; ++i) {
        if (res[i] < min_positive) {
            res[i] = min_positive;
        }
    }
    return res;
}
void _refine(const float* data, const float* vars, const int data_length,
    const char* mv_table, const int mv_length,
    const float* kmer_current, const int sequence_length,
    const int band_width, int* mv) {
    const int stride = mv_table[0];
    //build the matrix
    std::vector<int> coo_x;
    std::vector<int> coo_y;
    coo_x.reserve((band_width * 2 + 1) * data_length);
    coo_y.reserve((band_width * 2 + 1) * data_length);
    int seq_id = -1;
    int mv_id = 0;
    for (int i = 0; i < data_length; ++i) {
        if (i % stride == 0) {
            mv_id += 1;
            mv_id = (mv_id >= mv_length) ? (mv_length - 1) : mv_id;
            if (mv_table[mv_id] == 1) {
                seq_id += 1;
                seq_id = (seq_id >= sequence_length) ? (sequence_length - 1) : seq_id;
            }
        }
        if ((seq_id == -1) || (i == 0)) {
            coo_x.push_back(i);
            coo_y.push_back(0);
            continue;
        }
        for (int offset = -band_width; offset <= band_width; ++offset) {
            const int j = seq_id + offset;
            if ((j < 0) || (j >= sequence_length)) {
                continue;
            }
            coo_x.push_back(i);
            coo_y.push_back(j);
        }
    }
    //forward
    SparseMatrix<int> dp(coo_x.data(), coo_y.data(), coo_x.size(), data_length, sequence_length, -1);
    std::vector<float> pre_states;
    std::vector<float> cur_states;
    std::vector pre_j_col(sequence_length, -1);
    std::vector<float> state_item(2);
    std::vector<int> pre_j_item(2);
    SparseVectorView<int> dp_ptr = dp.get_row(0);
    pre_states.resize(dp_ptr.size);
    for (int j = 0; j < dp_ptr.size; ++j) {
        const int j_glob = dp_ptr.indices[j];
        const float y = kmer_current[j_glob];
        const float x = data[0];
        pre_states[j] = (x - y) * (x - y) / vars[0];
        pre_j_col[j_glob] = j;
    }
    for (int i = 1; i < data_length; ++i) {
        dp_ptr = dp.get_row(i);
        cur_states.resize(dp_ptr.size);
        const float x = data[i];
        const float N = i + 1;
        for (int j = 0; j < dp_ptr.size; ++j) {
            const int j_glob = dp_ptr.indices[j];
            const float y = kmer_current[j_glob];
            for (int p = 0; p <= 1; ++p) {
                const int j_glob_pre = j_glob - p;
                state_item[p] = std::numeric_limits<float>::max();
                pre_j_item[p] = -1;
                if ((j_glob_pre  < 0)) {
                    continue;
                }
                pre_j_item[p] = pre_j_col[j_glob_pre];
                const int _j = pre_j_item[p];
                if ((_j < 0) || (_j >= pre_states.size())) {
                    continue;
                }
                state_item[p] = pre_states[_j] + (x - y) * (x - y) / vars[i];
            }
            const int best_item = (state_item[0] <= state_item[1]) ? 0 : 1;
            cur_states[j] = state_item[best_item];
            dp_ptr.data[j] =  pre_j_item[best_item];
        }
        std::swap(cur_states, pre_states);
        std::ranges::fill(pre_j_col, -1);
        for (int j = 0; j < dp_ptr.size; ++j) {
            const int col = dp_ptr.indices[j];
            pre_j_col[col] = j;
        }
    }
    //backward
    std::fill_n(mv, data_length, -1);
    dp_ptr = dp.get_row(data_length - 1);
    int max_col = dp_ptr.size - 1;
    int last_j = 0;
    for (int i = data_length - 1; i >= 0; --i) {
        if (max_col < 0) {
            if (last_j != 0) {
                std::cout << "Refine: match chain break";
            }
            break;
        }
        SparseVectorView<int> dp_ptr = dp.get_row(i);
        last_j = dp_ptr.indices[max_col];
        mv[i] = dp_ptr.indices[max_col];
        max_col = dp_ptr.data[max_col];
    }
}

extern "C" {
void refine(
    float* data, const int data_length,
    const char* mv_table, const int mv_length,
    const char* sequence, const int sequence_length,
    const int band_width, const int iters, int* mv)
{
    constexpr int kmer = 9;
    constexpr int pad = 6;
    std::vector<float> kmer_current;
    kmer_current.reserve(sequence_length);
    const auto kmer_idx = kmer_to_index(sequence, sequence_length, kmer, pad);
    for (int i = 0; i < kmer_idx.size(); ++i) {
        kmer_current.push_back(kmer_current_map[kmer_idx[i]]);
    }
    std::vector<float> vars = sliding_vars(data, data_length, 11);
    for (int i = 0; i < iters; i++) {
        _refine(data, vars.data(), data_length, mv_table, mv_length, kmer_current.data(), sequence_length, band_width, mv);
        scale_signal(data, kmer_current.data(), mv, data_length);
    }
}
}