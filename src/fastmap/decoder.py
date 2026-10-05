# Copyright (c) 2026 Yuanjie Li. All rights reserved.
# Author: Yuanjie Li
# See LICENSE in the project root for terms.

import ctypes 
from itertools import product
import numpy as np 
from .bindings import get_library

def build_graph(kmer = 5, nbase = 4, alphabet = 'ACGT') -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    if kmer < 1 or not 2 <= nbase <= 4:
        raise ValueError('Require kmer >= 1 and 2 <= nbase <= 4')
    dim = nbase ** kmer
    words = lambda n: [''.join(p) for p in product(alphabet, repeat = n)]
    states = [w for w in words(kmer)]
    states = states
    state_ids = {s: i for i, s in enumerate(states)}
    nstate = len(states)
    edges = []  # (source, destination, kind). [RNA, ANOMALY, STAY];
    for dst in range(dim):
        w = states[dst]
        for a in alphabet:
            edges.append((state_ids[a + w[:-1]], dst, 'move'))
    edges += [(s, s, 'stay') for s in range(nstate)]
    alpahbet_id = {"A": 0, "C": 1, "G": 2, "T": 3}
    labels = [] 
    for _, dst, kind in edges:
        if kind == "stay":
            labels.append(-1) 
        elif kind == "move":
            labels.append(alpahbet_id[states[dst][-1]])
    return nstate, tuple(src for src, _, _ in edges), tuple(dst for _, dst, _ in edges), tuple(i for i in labels)

class GraphDecoder:
    def __init__(self, nstate, src, dst, labels):
        self.nstate = int(nstate)
        self.src = np.ascontiguousarray(src, dtype = np.int32)
        self.dst = np.ascontiguousarray(dst, dtype = np.int32)
        self.labels = np.ascontiguousarray(labels, dtype = np.int32)
        if self.src.ndim != 1 or self.src.size == 0 or self.dst.shape != self.src.shape or self.labels.shape != self.src.shape:
            raise ValueError('src/dst/labels must be equal-size nonempty 1D edge arrays')
        if not 0 < self.nstate < 2**31 or self.src.size >= 2**31:
            raise ValueError('graph dimensions exceed int32 limits')
        if np.any(self.src < 0) or np.any(self.src >= self.nstate) or np.any(self.dst < 0) or np.any(self.dst >= self.nstate) or np.any(self.labels < -1):
            raise ValueError('invalid graph indices or labels')
        self.lib = get_library()
        self.fn = self.lib.graph_beam_decode

    def decode(self, scores, lengths, alphabet = "ACGTD", beam_width = 32, beam_cut = 100.0, start = None, end = None):
        """Return (sequences, moves, probabilities), three lists of length B.

        sequences[b]: str, only emitted symbols (no initial context).
        moves[b]: uint8 [valid_T], 0=no emission, 1=one emitted symbol.
        probabilities[b]: float32 [len(sequences[b])], symbol posterior at
        its selected emission frame. Not a calibrated base accuracy/Q score.
        """
        if not isinstance(alphabet, str) or not alphabet or len(set(alphabet)) != len(alphabet):
            raise ValueError('alphabet must be a nonempty string of distinct symbols')
        rows = self.decode_detailed(scores, lengths, alphabet, beam_width, beam_cut, start, end)
        sequences, moves, probabilities = [], [], []
        for row in rows:
            mv = np.zeros(len(row['edges']), dtype=np.uint8)
            mv[row['token_times']] = 1
            sequences.append(row['sequence'])
            moves.append(mv)
            probabilities.append(row['token_confidence'])
        return sequences, moves, probabilities

    def decode_detailed(self, scores, lengths, alphabet = "ACGTD", beam_width = 32, beam_cut = 100.0, start = None, end = None):
        """Returns a list of per-read dictionaries, including posterior confidences.

        start/end: shared [S] log potentials; None means all states allowed.
        For masks use np.where(mask,0.,-np.inf). No initial symbols are emitted.
        The sequence log mass is approximate under beam pruning; logZ and
        time-local posterior confidences are computed on the full graph.
        """
        I = ctypes.POINTER(ctypes.c_int32)
        F = ctypes.POINTER(ctypes.c_float)
        D = ctypes.POINTER(ctypes.c_double)
        x = np.ascontiguousarray(scores, dtype=np.float32)
        if x.ndim != 3 or x.shape[2] != len(self.src):
            raise ValueError('scores must have shape [B,T,len(edges)]')
        B,T,E = x.shape
        if B < 1 or max(B,T,E) >= 2**31-1:
            raise ValueError('invalid score dimensions')
        if not isinstance(beam_width,(int,np.integer)) or not 1 <= beam_width < 2**31:
            raise ValueError('beam_width must be a positive int32')
        if lengths is None:
            lengths = np.full(B,T,dtype=np.int32)
        raw_lengths = np.asarray(lengths)
        if raw_lengths.shape != (B,) or np.any(raw_lengths < 0) or np.any(raw_lengths > T) or np.any(raw_lengths != np.floor(raw_lengths)):
            raise ValueError('lengths must contain B integers in [0,T]')
        lengths = np.ascontiguousarray(raw_lengths,dtype = np.int32)
        boundaries = []
        for values in (start, end):
            if values is None:
                boundaries.append(None)
            else:
                values = np.ascontiguousarray(values,dtype = np.float32)
                if values.shape != (self.nstate,):
                    raise ValueError('start/end must have shape [nstate]')
                boundaries.append(values)
        if alphabet is not None and any(i >= len(alphabet) for i in self.labels if i >= 0):
            raise ValueError('alphabet does not cover emitted token ids')
        tokens=np.full((B,T),-1,np.int32)
        times=np.full((B,T),-1,np.int32)
        token_conf=np.zeros((B,T),np.float32)
        out_len=np.zeros(B,np.int32)
        states=np.full((B,T+1),-1,np.int32)
        state_conf=np.zeros((B,T+1),np.float32)
        edges=np.full((B,T),-1,np.int32)
        edge_conf=np.zeros((B,T),np.float32)
        z=np.zeros(B,np.float64)
        mass=np.zeros(B,np.float64)
        best=np.zeros(B,np.float64)
        ptr=lambda a,t: None if a is None else a.ctypes.data_as(t)
        rc=self.fn(ptr(x,F),ptr(lengths,I),B,T,self.nstate,E,
                   ptr(self.src,I),ptr(self.dst,I),ptr(self.labels,I),
                   ptr(boundaries[0],F),ptr(boundaries[1],F),beam_width,beam_cut,
                   ptr(tokens,I),ptr(times,I),ptr(token_conf,F),ptr(out_len,I),
                   ptr(states,I),ptr(state_conf,F),ptr(edges,I),ptr(edge_conf,F),
                   ptr(z,D),ptr(mass,D),ptr(best,D))
        if rc:
            msg = self.lib.graph_decoder_last_error()
            raise RuntimeError(msg.decode() if msg else f'decode failed: {rc}')
        result=[]
        for b,n in enumerate(lengths):
            size=int(out_len[b]); n=int(n)
            row={
                'tokens': tokens[b,:size].copy(),
                'token_times': times[b,:size].copy(),
                'token_confidence': token_conf[b,:size].copy(),
                'states': states[b,:n+1].copy(),
                'state_confidence': state_conf[b,:n+1].copy(),
                'edges': edges[b,:n].copy(),
                'edge_confidence': edge_conf[b,:n].copy(),
                'log_partition': float(z[b]),
                'sequence_log_mass': float(mass[b]),
                'sequence_log_probability_lower_bound': float(mass[b]-z[b]),
                'path_log_score': float(best[b]),
            }
            if alphabet is not None:
                row['sequence']=''.join(alphabet[i] for i in row['tokens'])
            result.append(row)
        return result
