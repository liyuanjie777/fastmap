/*
 * Copyright (c) 2026 Yuanjie Li. All rights reserved.
 * Author: Yuanjie Li
 * See LICENSE in the project root for terms.
 */

#pragma once
#include <cmath>
#include <vector>
#include <algorithm>

inline void scale_signal(
    float* data,
    const float* reference,
    const int* mv,
    const int data_length
) {
    double sx = 0.0;
    double sy = 0.0;
    double sy2 = 0.0;
    double sxy = 0.0;
    const double N = data_length;
    //double residual = 0.0;
    for (int i = 0; i < data_length; ++i) {
        int j = mv[i];
        if (j < 0) {
            continue;
        }
        const double x = data[i];
        const double y = reference[j];
        sx += x;
        sy += y;
        sy2 += y * y;
        sxy += x * y;

        //residual += (x - y) * (x - y);
    }
    //std::cout << "residual" << sqrt(residual/N) << std::endl;
    if (N == 0) {
        return;
    }

    double scale, shift;
    if (const double denom = N * sy2 - sy * sy; std::abs(denom) < 1e-12) {
        scale = 1.0f;
        shift = sx / N - sy / N;
    }
    else {
        scale = (N * sxy - sx * sy) / denom;

        shift = (sx - scale * sy) / N;
    }
    for (int i = 0; i < data_length; ++i) {
        data[i] = (data[i] - shift) / scale;
    }
    return;
}

void inline coarse_align(int* mv, const int sequence_length, const int data_length, const int band_width, std::vector<int>& coo_x, std::vector<int>& coo_y) {
    if (data_length <= sequence_length) {
        return;
    }
    const int max_jump = data_length / sequence_length * 50;
    std::vector<int> helper;
    helper.reserve(sequence_length);
    coo_x.reserve(data_length * band_width);
    coo_y.reserve(data_length * band_width);
    int last_i = 0;
    for (int i = 0; i < data_length; ++i) {
        if (helper.empty()) {
            helper.push_back(mv[i]);
            last_i = i;
        }
        else if ((i - last_i) > max_jump) {
            helper.push_back(mv[i]);
            last_i = i;
        }
        else if (mv[i] > helper.back()) {
            helper.push_back(mv[i]);
            last_i = i;
        }
        mv[i] = helper.size() - 1;
    }
    const int bw = band_width - 1;
    for (int i = 0; i < data_length; ++i) {
        const int left = (mv[i] - 1 < 0) ? 0 : mv[i] - 1;
        const int right = (mv[i] + 1 >= helper.size()) ? helper.size() - 1 : mv[i] + 1;
        for (int j = helper[left] - bw; j <= helper[right] + bw; ++j) {
            if (j >= 0 && j < sequence_length) {
                coo_x.push_back(i);
                coo_y.push_back(j);
            }
        }
    }
}

inline std::vector<int> kmer_to_index(const char* seq, const int n, const int kmer, const int pad)
{
    static int char_to_bit_table[128] = {0};
    char_to_bit_table['A'] = char_to_bit_table['a'] = 0;
    char_to_bit_table['T'] = char_to_bit_table['t'] = 1;
    char_to_bit_table['C'] = char_to_bit_table['c'] = 2;
    char_to_bit_table['G'] = char_to_bit_table['g'] = 3;
    std::vector<int> codes;
    codes.reserve(n);
    if (n < kmer || kmer < pad) {
        return codes;
    }
    size_t  code = 0;
    for (int i = 0; i < kmer - pad; ++i) {
        code = (code << 2) | char_to_bit_table[static_cast<int>(seq[i]) & 127];
    }
    const size_t mask = (1LL << (2 * kmer)) - 1;
    codes.push_back(static_cast<int>(code));
    for (int i = kmer - pad; i < n; ++i) {
        code = ((code << 2) | char_to_bit_table[static_cast<int>(seq[i]) & 127]) & mask;
        codes.push_back(static_cast<int>(code));
    }
    for (int i = 0; i < kmer - pad - 1; ++i) {
        code = (code << 2) & mask;
        codes.push_back(static_cast<int>(code));
    }
    return codes;
}

inline int argmin(const float* data, const int n) {
    float min_val = std::numeric_limits<float>::max();
    int min_id = 0;
    for (int i = 0; i < n; ++i) {
        if (data[i] < min_val) {
            min_val = data[i];
            min_id = i;
        }
    }
    return min_id;
}

static inline float log_pdf_ig(const float t, const float mu, const float lambda)
{
    return 0.5f * std::log(lambda / (2.0f * M_PI * t * t * t))
         - (lambda * (t - mu) * (t - mu)) / (2.0f * mu * mu * t);
}
static inline float logaddexp(float a, float b)
{
    if (a == -INFINITY) return b;
    if (b == -INFINITY) return a;

    const float m = std::max(a, b);
    return m + std::log(std::exp(a - m) + std::exp(b - m));
}
static inline float log1mexp(float x)
{
    if (x >= 0.0f) return -INFINITY;
    if (x > -0.69314718056f) {
        return std::log(-std::expm1(x));
    } else {
        return std::log1p(-std::exp(x));
    }
}
inline std::vector<float> hazard_vector(const int R, const float jump_mean, const float jump_sigma) {
    std::vector<float> out(R * 2);
    const float mu = jump_mean;
    const float lambda = (mu * mu * mu) / (jump_sigma * jump_sigma);
    std::vector<float> log_pdf(R, -INFINITY);
    for (int r = 1; r < R; r++) {
        const float t = static_cast<float>(r);
        log_pdf[r] = log_pdf_ig(t, mu, lambda);
    }
    std::vector<float> logS(R, -INFINITY);
    float running = -INFINITY;
    for (int r = R - 1; r >= 1; r--) {
        running = (running == -INFINITY)
            ? log_pdf[r]
            : logaddexp(running, log_pdf[r]);

        logS[r] = running;
    }
    for (int r = 1; r < R; r++) {
        float log_jump = log_pdf[r] - logS[r];
        out[r * 2] = log_jump;
        out[r * 2 + 1] = log1mexp(log_jump);
    }
    return out;
}

inline float median(float* data, const int length) {
    if (length <= 0)
        return 0.0f;
    const int mid = length / 2;
    std::nth_element(data,data + mid,data + length);
    if (length & 1) {
        return data[mid];
    }
    const float a = data[mid];
    std::nth_element(data, data + mid - 1, data + length);
    return 0.5f * (a + data[mid - 1]);
}

inline std::vector<float> sliding_variance(const float* data, const int n, const int window) {
    std::vector<float> var(n);
    double sum = 0;
    double sum2 = 0;
    for(int i=0;i<n;i++) {
        sum += data[i];
        sum2 += data[i]*data[i];
        if(i >= window) {
            sum -= data[i-window];
            sum2 -= data[i-window]*data[i-window];
        }
        int count = std::min(i+1, window);
        double mean = sum / count;
        double v = sum2 / count - mean*mean;
        var[i] = std::max(0.0, v);
    }
    return var;
}

