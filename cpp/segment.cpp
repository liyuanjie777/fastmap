/*
 * Copyright (c) 2026 Yuanjie Li. All rights reserved.
 * Author: Yuanjie Li
 * See LICENSE in the project root for terms.
 */
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include "library.h"
#include "mymath.hpp"
#include "segment.h"

float _cusum_min_std(const float* data, const int n, const int window) {
    float mu = 0.0;
    float m2 = 0.0;
    for (int i = 0; i < window; i++) {
        float delta = data[i] - mu;
        mu += delta / static_cast<float>(i + 1);
        float delta2 = data[i] - mu;
        m2 += delta * delta2;
    }
    float min_var = m2;
    for (int i = window; i < n; i++) {
        float x_old = data[i - window];
        float x_new = data[i];
        float mu_old = mu;
        mu += (x_new - x_old) / static_cast<float>(window);
        m2 += (x_new - mu) * (x_new - mu_old) - (x_old - mu) * (x_old - mu_old);
        if (m2 < min_var) {
            min_var = m2;
        }
    }
    return std::sqrt(min_var / static_cast<float>(window));
}

extern "C" {
void cusum(const float* data, int* mv, const int data_size, const float sigma, const float h, const int window_size, const int min_segment_length) {
    if (data_size < 0 || window_size < 2 || min_segment_length < 1 ||
        !std::isfinite(sigma) || sigma <= 0 ||
        !std::isfinite(h) || h <= 0) {
        throw std::invalid_argument("Invalid CUSUM parameters");
        }
    if (data_size == 0) return;
    if (!data || !mv) throw std::invalid_argument("Null data or mv");
    for (int i = 0; i < data_size; ++i)
        if (!std::isfinite(data[i]))
            throw std::invalid_argument("Signal contains NaN or Inf");
    if (data_size == 1) { mv[0] = 0; return; }

    const float min_std = _cusum_min_std(data, data_size, window_size);
    const float shift = sigma;
    float gpos = 0.0, gneg = 0.0;
    float m1 = data[0], m2 = 0.0;
    int anchor = 0, idx = 0;
    int pos_start = 1, neg_start = 1;
    mv[0] = 0;

    for (int i = 1; i < data_size; ++i) {
        const float delta = data[i] - m1;
        m1 += delta / (i - anchor + 1);
        m2 += delta * (data[i] - m1);
        const float stddev = std::max(min_std,
            std::sqrt(std::max(0.0f, m2 / (i - anchor + 1))));
        const double z = (data[i] - m1) / stddev;

        if (gpos <= 0.0) pos_start = i;
        if (gneg <= 0.0) neg_start = i;
        gpos = std::max(0.0, gpos + shift * z - shift * shift / 2);
        gneg = std::max(0.0, gneg - shift * z - shift * shift / 2);

        if (gpos >= h || gneg >= h) {
            if ((i - anchor + 1) / 2 >= min_segment_length) {
                const int candidate = gpos >= gneg ? pos_start : neg_start;
                const int cut = std::max(candidate, anchor + min_segment_length);
                const int latest_cut = i + 1 - min_segment_length;
                if (cut <= latest_cut) {
                    ++idx;
                    for (int j = cut; j <= i; ++j) mv[j] = idx;
                    anchor = cut;
                    m1 = 0.0;
                    m2 = 0.0;
                    for (int j = cut; j <= i; ++j) {
                        const float d = data[j] - m1;
                        m1 += d / (j - cut + 1);
                        m2 += d * (data[j] - m1);
                    }
                    gpos = gneg = 0.0;
                    pos_start = neg_start = i + 1;
                }
            }
        }
        mv[i] = idx;
    }
}

void pelt(const float* data, int* mv, const int data_size, const float penalty, const int min_segment_length) {
    if (data_size < 0 || !std::isfinite(penalty) || penalty < 0)
        throw std::invalid_argument("invalid size or penalty");
    if (data_size == 0) return;
    const int min_size = std::max(min_segment_length, 2);
    if (!data || !mv || data_size < min_size)
        throw std::invalid_argument("null pointer or signal shorter than min_size");

    const std::size_t n = static_cast<std::size_t>(data_size);
    const std::size_t minimum = static_cast<std::size_t>(min_size);
    std::vector<long double> sum(n + 1, 0), squares(n + 1, 0);
    const long double shift = data[0];
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(data[i]))
            throw std::invalid_argument("non-finite signal");
        const long double y = static_cast<long double>(data[i]) - shift;
        sum[i + 1] = sum[i] + y;
        squares[i + 1] = squares[i] + y * y;
    }

    std::vector<double> dp(n + 1, std::numeric_limits<double>::infinity());
    std::vector<std::size_t> previous(n + 1), candidates;
    std::vector<double> values;
    dp[0] = 0;
    for (std::size_t b = minimum; b <= n; ++b) {
        const std::size_t new_start = b - minimum;
        if (std::isfinite(dp[new_start])) candidates.push_back(new_start);
        values.clear();
        for (const std::size_t a : candidates) {
            const long double length = b - a;
            const long double mean = (sum[b] - sum[a]) / length;
            long double variance = (squares[b] - squares[a]) / length - mean * mean;
            const long double scale = (squares[b] + squares[a]) / length + mean * mean;
            const long double bound = 64 * std::numeric_limits<long double>::epsilon()
                                      * static_cast<long double>(b + 1) * scale;
            if (variance < 0 || bound > 1e-10L * (std::max(variance, 0.0L) + 1e-6L)) {
                // Stable local two-pass fallback when prefix subtraction loses precision.
                const long double base = data[a];
                long double local_mean = 0;
                for (std::size_t i = a; i < b; ++i)
                    local_mean += static_cast<long double>(data[i]) - base;
                local_mean /= length;
                variance = 0;
                for (std::size_t i = a; i < b; ++i) {
                    const long double d = (static_cast<long double>(data[i]) - base) - local_mean;
                    variance += d * d;
                }
                variance /= length;
            }
            const double cost = static_cast<double>(length)
                              * std::log(static_cast<double>(variance) + 1e-6);
            const double value = dp[a] + (cost + static_cast<double>(penalty));
            values.push_back(value);
            if (value < dp[b]) { // First candidate wins exact ties.
                dp[b] = value;
                previous[b] = a;
            }
        }
        // Same immediate pruning as ruptures; retains its min_size limitations.
        std::size_t kept = 0;
        for (std::size_t j = 0; j < candidates.size(); ++j)
            if (values[j] <= dp[b] + static_cast<double>(penalty))
                candidates[kept++] = candidates[j];
        candidates.resize(kept);
    }

    int segment_count = 0;
    for (std::size_t b = n; b > 0; b = previous[b]) ++segment_count;
    for (std::size_t b = n; b > 0;) {
        const std::size_t a = previous[b];
        --segment_count;
        std::fill(mv + a, mv + b, segment_count);
        b = a;
    }
}
}
