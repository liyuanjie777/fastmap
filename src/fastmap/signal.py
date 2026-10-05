# Copyright (c) 2026 Yuanjie Li. All rights reserved.
# Author: Yuanjie Li
# See LICENSE in the project root for terms.


# src/fastmap/signal.py
import numpy as np
from .bindings import FloatPtr, IntPtr, CharPtr, get_library

def cusum(data, sigma = 1.0, h = 3.0, window_size = 12, min_window_length = 12):
    data = np.array(data, dtype = np.float32, order = "C", copy = True)
    mv = np.zeros(data.size, dtype = np.int32)
    lib = get_library()
    lib.cusum(data.ctypes.data_as(FloatPtr), mv.ctypes.data_as(IntPtr), data.size, sigma, h, window_size, min_window_length)
    return data, mv
    
def pelt(data, penalty = 10.0, min_window_length = 12):
    data = np.array(data, dtype = np.float32, order = "C", copy = True)
    mv = np.zeros(data.size, dtype = np.int32)
    lib = get_library()
    lib.pelt(data.ctypes.data_as(FloatPtr), mv.ctypes.data_as(IntPtr), data.size, penalty, min_window_length)
    return data, mv
    
def refine(data, mv_table, sequence, niter = 10, band_width = 3):
    data_length = len(data)
    data = np.array(data, np.float32, copy = True)
    mv_table = np.array(mv_table, dtype = np.int8, copy = True)
    mv = np.zeros(data.size, dtype = np.int32)
    sequence_bytes = sequence.encode("ascii")
    sequence_length = int(np.count_nonzero(mv_table == 1))
    lib = get_library()
    lib.refine(data.ctypes.data_as(FloatPtr), data.size, mv_table.ctypes.data_as(CharPtr), mv_table.size, sequence_bytes, sequence_length, band_width, niter, mv.ctypes.data_as(IntPtr))
    return data, mv

def mv2mvidx(mv, data_length):
    """convert dorado move table to the move idx for sequence
    """
    stride = int(mv[0])
    res = np.zeros(stride * (len(mv) - 1), np.int32)
    res = np.array(mv[1:], np.int32)
    res = np.cumsum(res)
    res = res - res[0]
    res = np.repeat(res, stride)
    if res.size < data_length:
        res = np.pad(res, (0, data_length - res.size), 'edge')
    res = res[ : data_length]
    return res

def mvcut(data, mv_table):
    mv = mv2mvidx(mv_table, len(data))
    return data, mv

