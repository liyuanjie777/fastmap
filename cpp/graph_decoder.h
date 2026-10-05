/*
 * Copyright (c) 2026 Yuanjie Li. All rights reserved.
 * Author: Yuanjie Li
 * See LICENSE in the project root for terms.
 */
#ifndef FASTMAP_GRAPH_DECODER_H
#define FASTMAP_GRAPH_DECODER_H
#include <stdint.h>
extern "C" {
int graph_beam_decode(
    const float* scores, const int32_t* lengths, int32_t B, int32_t T,
    int32_t S, int32_t E, const int32_t* src, const int32_t* dst,
    const int32_t* labels, const float* start, const float* end,
    int32_t beam_width, double beam_cut,
    int32_t* tokens, int32_t* token_times, float* token_conf,
    int32_t* out_lengths, int32_t* states, float* state_conf,
    int32_t* path_edges, float* edge_conf, double* log_z,
    double* sequence_log_mass, double* path_log_score);
const char* graph_decoder_last_error(void);
}


#endif // FASTMAP_GRAPH_DECODER_H
