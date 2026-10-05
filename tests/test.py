# Copyright (c) 2026 Yuanjie Li. All rights reserved.
# Author: Yuanjie Li
# See LICENSE in the project root for terms.


import argparse
import traceback
from pathlib import Path
import numpy as np
from fastmap import GraphDecoder, cusum, pelt, refine, build_graph, showCurrent, mvcut

def load_sample(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Sample not found: {path}')
    with np.load(path, allow_pickle=False) as stored:
        result = {name: stored[name] for name in stored.files}
    required = {'signal', 'move_table', 'query', 'ref', 'score'}
    missing = required - result.keys()
    if missing:
        raise ValueError(f'Missing sample fields: {sorted(missing)}')
    for name in ('query', 'ref'):
        if result[name].ndim != 0:
            raise ValueError(f'{name} must be a scalar string')
        result[name] = result[name].item()
        if not isinstance(result[name], str):
            raise ValueError(f'{name} must be a string')
    return result


def test_signal_algorithms(sample, function):
    signal = sample['signal'].copy()
    before = signal.copy()
    data, moves = function(signal)
    showCurrent(data, moves, ishow = True)
    assert data.shape == signal.shape
    assert moves.shape == signal.shape
    assert data.dtype == np.float32
    assert moves.dtype == np.int32
    assert np.isfinite(data).all()
    np.testing.assert_array_equal(signal, before)
    key = f'expected_{function.__name__}_moves'
    if key in sample:
        np.testing.assert_array_equal(moves, sample[key])

def test_mv_sample(sample):
    signal, mv = sample['signal'], sample['move_table']
    signal_copy, mv_copy = signal.copy(), mv.copy()
    data, moves = mvcut(signal_copy, mv_copy)
    showCurrent(data, moves, ishow = True)

def test_refine_sample(sample):
    signal, mv, query = sample['signal'], sample['move_table'], sample['query']
    assert np.count_nonzero(mv == 1) <= len(query.encode('ascii'))
    signal_copy, mv_copy = signal.copy(), mv.copy()
    data, moves = refine(signal_copy, mv_copy, query)
    showCurrent(data, moves, ishow = True)
    assert data.shape == signal.shape
    assert moves.shape == signal.shape
    assert data.dtype == np.float32
    assert moves.dtype == np.int32
    assert np.isfinite(data).all()
    np.testing.assert_array_equal(signal_copy, signal)
    np.testing.assert_array_equal(mv_copy, mv)
    if 'expected_refine_moves' in sample:
        np.testing.assert_array_equal(moves, sample['expected_refine_moves'])


def test_decoder_sample(sample):
    alphabet = "ACGT"
    graph_spec = build_graph()
    score = sample['score']
    decoder = GraphDecoder(*graph_spec)
    sequences, moves, probabilities = decoder.decode(score[None, :, :], [score.shape[0]], alphabet = alphabet)
    print(f">decoder\n{sequences[0]}\n>query\n{sample['query']}\n>reference\n{sample['ref']}")

def main():
    parser = argparse.ArgumentParser(description='Check fastmap using a saved sample')
    sample_path = str(Path(__file__).resolve().parent / 'exmple.npz')
    args = parser.parse_args()
    sample = load_sample(sample_path)
    checks = [
        ('cusum', lambda: test_signal_algorithms(sample, cusum)),
        ('pelt', lambda: test_signal_algorithms(sample, pelt)),
        ('mvcut', lambda: test_mv_sample(sample)),
        ('refine', lambda: test_refine_sample(sample)),
    ]
    checks.append(('GraphDecoder', lambda: test_decoder_sample(sample)))

    passed, failed = 1, 0
    for name, run in checks:
        print(f'[RUN] {name}', flush=True)
        try:
            run()
        except Exception:
            failed += 1
            print(f'[FAIL] {name}', flush=True)
            traceback.print_exc()
        else:
            passed += 1
            print(f'[PASS] {name}', flush=True)
    print(f'Passed: {passed}, failed: {failed}')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
