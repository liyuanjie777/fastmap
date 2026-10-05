/*
 * Copyright (c) 2026 Yuanjie Li. All rights reserved.
 * Author: Yuanjie Li
 * See LICENSE in the project root for terms.
 */

#pragma once

extern "C" {
void cusum(const float* data, int* mv, const int data_size, const float sigma, const float h, const int window_size, const int min_segment_length);
void pelt(const float* data, int* mv, const int data_size, const float penalty, const int min_segment_length);
}