/*
 * Copyright (c) 2026 Yuanjie Li. All rights reserved.
 * Author: Yuanjie Li
 * See LICENSE in the project root for terms.
 */

#ifndef FASTMAP_LIBRARY_H
#define FASTMAP_LIBRARY_H

#include <array>
#include <cstdint>
#include <ext/numeric_traits.h>
#include <list>
#include <stdexcept>
#include <vector>

constexpr int kernal = 6;
struct State {
    float cost;
    float run;
};

template <typename T>
class SparseVectorView {
public:
    SparseVectorView(): data(nullptr), indices(nullptr), size(0) {};
    T* data;
    const int* indices;
    int size;
};

template <typename T>
class SparseMatrix {
public:
    SparseMatrix(const int* x, const int* y, const int num, const int row, const int col, const T& val):
    _row(row), _col(col), _num(num), _val(num, val), _indices_csr(num, 0), _ptr_csr(row + 1, 0) {
        for (int i = 0; i < num; ++i)
        {
            _indices_csr[i] = y[i];
            _ptr_csr[x[i] + 1]++;
        }
        for (int i = 0; i < _row; ++i)
        {
            _ptr_csr[i + 1] += _ptr_csr[i];
        }
    };
    SparseMatrix(): _row(0), _col(0), _num(0) {};
    SparseVectorView<T> get_row(const int i) {
        if (i >= _row) {
            throw std::out_of_range("index out of range");
        }
        SparseVectorView<T> result;
        result.data = &_val[_ptr_csr[i]];
        result.indices = &_indices_csr[_ptr_csr[i]];
        result.size = _ptr_csr[i + 1] - _ptr_csr[i];
        return result;
    }

private:
    int _row;
    int _col;
    size_t _num;
    std::vector<T> _val;
    std::vector<int> _indices_csr;
    std::vector<int> _ptr_csr;
};

extern "C" {
void refine(
    float* data, const int data_length,
    const char* mv_table, const int mv_length,
    const char* sequence, const int sequence_length,
    const int band_width, const int iters, int* mv);
}
#endif // FASTMAP_LIBRARY_H
