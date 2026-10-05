# fastmap

fastmap provides Python interfaces for nanopore signal processing, sequence-guided signal refinement, and graph decoding. Computation runs in a C++ shared library, accessed through ctypes with NumPy arrays.

The current build configuration targets Linux and bundles `libfastmap.so` with the Python package. This README describes the original C++ decoder interface with its full set of outputs. It does not describe the alternative simplified ABI.

## Project layout

```text
fastmap-project/
├── pyproject.toml
├── setup.py
├── README.md
├── LICENSE
├── cpp/
│   ├── CMakeLists.txt
│   ├── *
├── src/
│   └── fastmap/
│       ├── __init__.py
│       ├── bindings.py
│       ├── signal.py
│       ├── decoder.py
│       ├── model.py
│       └── release/
│           └── libfastmap.so
└── tests/
    ├── test_sample.py
    └── exmple.npz
```

## Build and install

Requirements include Python 3.10 or later, NumPy, CMake, and a compiler supporting C++17. Additional C++ dependencies are defined in `CMakeLists.txt`. The minimum Python and NumPy versions declared in the package configuration should be tested before release.

Download and Run these commands from the project root:

```bash
python -m pip install .
```

- `pip install .`: installs the Python package.

## Public API

```python
from fastmap import cusum, pelt, refine, GraphDecoder, model_convert
```

| Interface | Purpose |
| --- | --- |
| `cusum` | Calls the C++ CUSUM signal-processing routine |
| `pelt` | Calls the C++ PELT signal-segmentation routine |
| `refine` | Refines a signal using a move table and a sequence |
| `GraphDecoder.decode` | Returns sequences, per-frame moves, and symbol probabilities |
| `GraphDecoder.decode_detailed` | Returns paths, posterior confidences, and log scores |

The signal-processing signatures and defaults below follow the Python wrappers. The precise boundary rules, output move-table encoding, and signal preprocessing requirements for `cusum`, `pelt`, and `refine` are defined by their C++ implementations. Their move tables should not automatically be interpreted as the graph decoder's binary moves.

## cusum

```python
cusum(data, sigma=1.0, h=3.0, window_size=12, min_window_length=12)
```

| Parameter | Description |
| --- | --- |
| `data` | A one-dimensional signal; copied into a float32 array before the C++ call |
| `sigma` | CUSUM scale parameter passed to C++; default: 1.0 |
| `h` | CUSUM threshold parameter; default: 3.0 |
| `window_size` | Window-size parameter; default: 12 |
| `min_window_length` | Minimum window-length parameter; default: 12 |

Returns `(data, mv)`. `data` is the float32 copy after C++ processing. `mv` is an int32 output array with the same length as the input signal. The caller's original signal should remain unchanged.

```python
import numpy as np
from fastmap import cusum

with np.load("tests/data/exmple.npz", allow_pickle=False) as sample:
    signal = sample["signal"]

data, mv = cusum(signal, sigma=1.0, h=3.0,
                 window_size=12, min_window_length=12)
```

## pelt

```python
pelt(data, penalty=10.0, min_window_length=12)
```

| Parameter | Description |
| --- | --- |
| `data` | A one-dimensional signal, copied into a float32 array |
| `penalty` | Segmentation penalty; default: 10.0. The exact cost function is defined by the C++ implementation |
| `min_window_length` | Minimum window-length parameter; default: 12 |

Returns `(data, mv)` with the same type and shape conventions as `cusum`.

```python
import numpy as np
from fastmap import pelt

with np.load("tests/data/exmple.npz", allow_pickle=False) as sample:
    signal = sample["signal"]

data, mv = pelt(signal, penalty=10.0, min_window_length=12)
```

## refine

```python
refine(data, mv_table, sequence, niter=10, band_width=3)
```

| Parameter | Description |
| --- | --- |
| `data` | A one-dimensional raw or preprocessed signal, as required by the C++ model |
| `mv_table` | A one-dimensional move table corresponding to the signal and sequence; converted to int8 |
| `sequence` | An ASCII sequence string, typically the query associated with the move table |
| `niter` | Iteration-count parameter; default: 10 |
| `band_width` | Search-band-width parameter; default: 3 |

Returns `(data, mv)`: a float32 signal copy and an int32 output move table with the same length as the signal.

```python
import numpy as np
from fastmap import refine

with np.load("tests/data/exmple.npz", allow_pickle=False) as sample:
    signal = sample["signal"]
    move_table = sample["move_table"]
    query = sample["query"].item()

data, mv = refine(signal, move_table, query, niter=10, band_width=3)
```

The current wrapper passes `count(mv_table == 1)` as the sequence length to C++. Inputs must follow this convention, and that length must not exceed the actual ASCII string buffer. If a move table contains a stride header, allows moves greater than one, or comes from another tool, convert it to the format required by the C++ interface first.

`ref` is not a separate parameter of `refine`. A reference sequence can only be used as `sequence` when its correspondence to the signal and move table is appropriate; it cannot be substituted for the query arbitrarily.

This README uses the parameter name `niter`. If your local wrapper still uses `iter`, use the keyword matching that signature or make the naming consistent first.

## GraphDecoder

### Construct a graph

```python
graph_spec = build_graph()
decoder = GraphDecoder(*graph_spec)
```

| Parameter | Description |
| --- | --- |
| `nstate` | Number of states S; a positive integer |
| `src` | An int32 array of length E; source state of each edge |
| `dst` | An int32 array of length E; destination state of each edge |
| `labels` | An int32 array of length E; nonnegative values identify emitted symbols, and -1 means no emission |

State indices must be in `[0, S)`. The three edge arrays must be nonempty, one-dimensional, and equal in length. Edge e corresponds to `scores[..., e]`, so concatenating or reordering score columns requires a matching graph edge order.

Every edge consumes one frame. A label of -1 suppresses symbol emission but still consumes a frame. The initial state does not automatically emit sequence characters.

### decode

```python
decoder.decode(scores, lengths, alphabet="ACGTD", beam_width=32,
               beam_cut=100.0, start=None, end=None)
```

| Parameter | Description |
| --- | --- |
| `scores` | float32 `[B, T, E]`; per-frame, per-edge log scores or log potentials |
| `lengths` | An integer array `[B]` with valid frame counts in `[0, T]`; pass None to use all T frames |
| `alphabet` | A nonempty string of distinct characters; `alphabet[label]` gives the emitted character |
| `beam_width` | Maximum number of candidates retained per frame; default: 32 |
| `beam_cut` | Nonnegative pruning threshold relative to the best candidate rank; default: 100.0 |
| `start/end` | Log-weight arrays `[S]`, or None; shared across the batch |

Scores are interpreted as log scores. Probabilities are not automatically converted to logs. Negative infinity is allowed to forbid an edge or boundary state; NaN and positive infinity are not allowed.

When `start/end` are None, every boundary state has weight zero. Negative infinity in a boundary array forbids that state. A path's total score is its start weight plus its per-frame edge scores plus its end weight.

Returns three lists, each of length B:

| Return value | Contents of each batch item |
| --- | --- |
| `sequences[b]` | The decoded string |
| `moves[b]` | uint8 `[lengths[b]]`; 1 means a character was emitted at that frame, and 0 means none was emitted |
| `probabilities[b]` | float32 `[len(sequences[b])]`; symbol posterior at each selected emission frame |

Symbol probabilities sum over edges with the same label on the full graph. They are not base accuracy estimates or calibrated Phred Q scores. Decoding selects a sequence by its retained beam mass, then uses a representative path for that sequence to determine emission frames.

This example runs independently:

```python
import numpy as np
from fastmap import GraphDecoder

# One state, with two self-loop edges emitting A and C.
decoder = GraphDecoder(1, src=[0, 0], dst=[0, 0], labels=[0, 1])
scores = np.log(np.array([[[0.8, 0.2]]], dtype=np.float32))

sequences, moves, probabilities = decoder.decode(
    scores, lengths=[1], alphabet="AC"
)
print(sequences)         # ['A']
print(moves[0])          # [1]
print(probabilities[0])  # Approximately [0.8]
```

For stored scores with shape `[T, E]`, add a batch dimension:

```python
# score: [T, E]; the decoder graph must match the score columns.
sequences, moves, probabilities = decoder.decode(
    score[None, :, :], lengths=[score.shape[0]], alphabet="ACGTD"
)
```

Example boundary constraints:

```python
# For a graph with nstate == 3, allow only state 0 as the start
# and state 2 as the end.
start = np.array([0, -np.inf, -np.inf], dtype=np.float32)
end = np.array([-np.inf, -np.inf, 0], dtype=np.float32)
# Pass start=start and end=end to decode.
# The graph must contain a path satisfying these constraints.
```

### decode_detailed

```python
decoder.decode_detailed(scores, lengths, alphabet="ACGTD", beam_width=64,
                        beam_cut=100.0, start=None, end=None)
```

Parameters follow `decode`, except that the default beam width is 64. Explicitly use the same beam width when comparing the two methods. When `alphabet=None`, token IDs are returned without a `sequence` field.

Returns one dictionary per read. Let N be the valid frame count and L the emitted sequence length:

| Field | Type / shape | Description |
| --- | --- | --- |
| `sequence` | str | Decoded string; omitted when alphabet is None |
| `tokens` | int32 `[L]` | Emitted symbol IDs |
| `token_times` | int32 `[L]` | Corresponding zero-based frame indices |
| `token_confidence` | float32 `[L]` | Symbol posteriors at the selected frames |
| `states` | int32 `[N+1]` | Representative path states, including the initial state before the first frame |
| `state_confidence` | float32 `[N+1]` | Posteriors of the corresponding states |
| `edges` | int32 `[N]` | Selected edge at each frame |
| `edge_confidence` | float32 `[N]` | Posteriors of the selected edges |
| `log_partition` | float | Log total mass of all allowed paths on the full graph |
| `sequence_log_mass` | float | Selected sequence's log mass from beam-retained paths |
| `sequence_log_probability_lower_bound` | float | sequence_log_mass minus log_partition |
| `path_log_score` | float | Log score of the representative path for the selected sequence |

```python
rows = decoder.decode_detailed(scores, [1], alphabet="AC", beam_width=32)
print(rows[0]["sequence"])
print(rows[0]["token_times"])
print(rows[0]["log_partition"])
```

This example uses the single-state graph and scores from the independent example above. If no path is available or the C++ routine reports an error, the wrapper raises RuntimeError rather than returning usable results.

## Test data and execution

The sample test script runs with ordinary Python and does not require pytest:

```bash
python tests/test_sample.py
```

Sample fields:

| Field | Format |
| --- | --- |
| `signal` | float32 `[N]` |
| `move_table` | A one-dimensional int8 array following the refine input convention |
| `query` / `ref` | Strings; retrieve them using `.item()` |
| `score` | float32 `[T, E]` log scores |
| `nstate/src/dst/labels/alphabet` | Optional graph metadata, required for decoding the stored scores |
| `start/end` | Optional boundary log weights |

To save prepared data:

```python
np.savez_compressed(
    "exmple.npz",
    signal=np.asarray(signal, dtype=np.float32),
    move_table=np.asarray(mv, dtype=np.int8),
    query=que,
    ref=ref,
    score=np.asarray(score, dtype=np.float32),
)
```

Do not concatenate scores again if they already have the intended `[T, E]` layout. For decoding, also save graph metadata matching the final column order.

The script checks fields, successful execution, output shapes, and probability ranges. It explicitly skips GraphDecoder when graph metadata is missing. Optional fields `expected_cusum_moves`, `expected_pelt_moves`, `expected_refine_moves`, and `expected_sequence` enable comparison with trusted expected results.

Structural checks do not establish algorithmic accuracy. Query and reference strings are not automatically treated as ground truth. Reference-based accuracy evaluation requires a separate definition of alignment direction, interval, and metrics.

## License

All rights reserved. This project does not grant an open-source license. Except as permitted by applicable law or separate written authorization, use, copying, modification, and redistribution are not permitted. See [LICENSE](LICENSE) for the full notice. Third-party components remain subject to their respective licenses.
