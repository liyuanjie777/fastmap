# Copyright (c) 2026 Yuanjie Li. All rights reserved.
# Author: Yuanjie Li
# See LICENSE in the project root for terms.


#src/fastmap/bindings.py
import ctypes
from functools import lru_cache
from pathlib import Path 

FloatPtr = ctypes.POINTER(ctypes.c_float)
IntPtr = ctypes.POINTER(ctypes.c_int)
Int32Ptr = ctypes.POINTER(ctypes.c_int32)
DoublePtr = ctypes.POINTER(ctypes.c_double)
CharPtr = ctypes.POINTER(ctypes.c_char)

@lru_cache(maxsize = 1)
def _load_library():
    path = Path(__file__).resolve().parent / "release" / "libfastmap.so" 
    if not path.is_file():
        raise FileNotFoundError("can not find libfastmap.so, compile and added into _release")
    return ctypes.CDLL(str(path))
    
@lru_cache(maxsize=1)
def get_library():
    lib = _load_library()
    lib.cusum.argtypes = [
        FloatPtr,
        IntPtr,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.cusum.restype = None
    lib.pelt.argtypes = [
        FloatPtr,
        IntPtr,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
    ]
    lib.pelt.restype = None
    lib.refine.argtypes = [
        FloatPtr,
        ctypes.c_int,
        CharPtr,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        IntPtr,
    ]
    lib.refine.restype = None
    lib.graph_beam_decode.argtypes = [
        FloatPtr, Int32Ptr,
        ctypes.c_int32, ctypes.c_int32,
        ctypes.c_int32, ctypes.c_int32,
        Int32Ptr, Int32Ptr, Int32Ptr,
        FloatPtr, FloatPtr,
        ctypes.c_int32, ctypes.c_double,
        Int32Ptr, Int32Ptr, FloatPtr, Int32Ptr,
        Int32Ptr, FloatPtr, Int32Ptr, FloatPtr,
        DoublePtr, DoublePtr, DoublePtr,
    ]
    lib.graph_beam_decode.restype = ctypes.c_int
    lib.graph_decoder_last_error.argtypes = []
    lib.graph_decoder_last_error.restype = ctypes.c_char_p
    return lib
