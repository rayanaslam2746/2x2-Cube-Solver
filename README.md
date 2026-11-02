# 2x2-Cube-Solver

A PyTorch 2x2 Rubik's Cube solver that learns with zero human knowledge.
It scrambles a solved cube and trains a network to predict the move that
undoes the last scramble move, then uses search to turn that policy into
full solutions. This is the same idea as
[EfficientCube (Takano, 2023)](https://arxiv.org/abs/2106.03157).

The 2x2 is small enough (3,674,160 states with one corner fixed) to compute
the exact optimal distance of every state, so every prediction is graded
against ground truth.

## Contents

- [Setup](#setup)
- [Quick start](#quick-start)
- [Running the tests](#running-the-tests)
- [Training](#training)
- [Evaluation](#evaluation)
- [Solving cubes from Python](#solving-cubes-from-python)
- [Files the program creates](#files-the-program-creates)
- [How it works](#how-it-works)

## Setup

Requires Python 3.9 or newer (developed on 3.13).

```bash
pip install -r requirements.txt
```

This installs `numpy`, `torch` and `pytest`. A GPU is used automatically when
PyTorch can see one. For an NVIDIA GPU, install a CUDA build of PyTorch from
[pytorch.org](https://pytorch.org/get-started/locally/). Everything also runs
on CPU, more slowly.

Run every command from the repository root so the `models/` paths resolve.

## Quick start

```bash
python -m pytest                                    # check everything works
python -m cube_solver.train --steps 10000           # train, saves models/policy.pt
python -m cube_solver.evaluate --methods greedy beam:256 astar:0.8   # grade it
```

## Running the tests

```bash
python -m pytest                              # everything
python -m pytest tests/test_cube.py           # simulator physics only (fast)
python -m pytest tests/test_state_space.py    # distance tables and solvers (builds tables, slower)
python -m pytest -k oracle                    # only tests whose names contain "oracle"
python -m pytest -v                           # list each test as it runs
```

The tests check four things:
- **The simulator** follows the rules of a real cube: every move is a
  permutation, a move followed by its inverse returns to the start, opposite
  faces commute, and there are 24 whole-cube rotations.
- **The distance tables** match the published number of states at each
  distance.
- **State indexing** round-trips: numbering a state and decoding it gives the
  same state back.
- **Every search** solves cubes in any orientation when guided by the exact
  table.

## Training

```bash
python -m cube_solver.train [flags]
python -m cube_solver.train --help
```

On the first run, the exact distance table is built (about 12 s) and the
frozen validation and test sets are sampled. Both are cached in `models/` and
reused by every later run. Each training step generates a fresh batch of
scrambled cubes, so there is no dataset to download or prepare.

> **Warning:** the default output is `models/policy.pt`. A new run overwrites
> it at its first validation check and rewrites `models/policy.csv`. Pass
> `--out` to keep an existing model.

### Training flags

| Flag | Default | What it does |
| --- | --- | --- |
| `--steps` | `5000` | Number of training steps. Each step uses a newly generated batch. |
| `--batch-size` | `1024` | Cubes per step. |
| `--lr` | `0.001` | Peak learning rate. A one-cycle schedule warms up to it, then decays to near zero by the last step. |
| `--weight-decay` | `1e-05` | AdamW weight decay. |
| `--labels` | `scramble` | `scramble` labels each state with the inverse of the last scramble move (no prior knowledge). `exact` samples random states and spreads the target over every truly optimal move from the distance table. |
| `--encoding` | `onehot` | Network input. `onehot` is 144 inputs (24 stickers x 6 colors). `pieces` is 70 inputs (which corner sits in each of the 7 movable slots, plus its twist). |
| `--half-turns` | off | Adds U2, R2 and F2 as single moves (half-turn metric). Uses its own distance table and held-out set. |
| `--max-depth` | `0` | Longest scramble. `0` means 20 (quarter turns) or 16 (half turns). Ignored with `--labels exact`. |
| `--curriculum-steps` | `0` | Grows scramble depth linearly from 1 up to `--max-depth` over this many steps. `0` means full depth from the start. |
| `--label-smoothing` | `0.0` | Softens the policy targets, e.g. `0.1`. |
| `--value-weight` | `0.1` | Weight of the value loss (predicted moves left) relative to the policy loss. |
| `--stem-width` | `1024` | Width of the first hidden layer. |
| `--width` | `512` | Width of the residual blocks. |
| `--n-blocks` | `3` | Number of residual blocks. |
| `--eval-every` | `250` | Validate, log and possibly save every N steps. |
| `--seed` | `0` | Random seed for weights and data. |
| `--out` | `models/policy.pt` | Checkpoint path. The training log is written next to it with a `.csv` extension. |
| `--resume` | *(none)* | Continue training from an existing checkpoint. See [Resuming](#resuming-a-run). |

### Full example

All flags written out explicitly. Except for `--steps` and `--out`, these are
the default values.

```bash
python -m cube_solver.train \
    --steps 1000 \
    --batch-size 1024 \
    --lr 1e-3 \
    --weight-decay 1e-5 \
    --labels scramble \
    --encoding onehot \
    --max-depth 20 \
    --curriculum-steps 0 \
    --label-smoothing 0.0 \
    --value-weight 0.1 \
    --stem-width 1024 \
    --width 512 \
    --n-blocks 3 \
    --eval-every 250 \
    --seed 0 \
    --out models/demo.pt
```

On Windows PowerShell, put it on one line or replace each `\` with a backtick.

Output (on a GPU; times vary by machine):

```
2,262,023 parameters on cuda; logging to models\demo.csv
step    250 | depth 20 | loss 1.6391 | label acc 0.413 | val optimal 0.738 | value MAE 2.74 | 4s
step    500 | depth 20 | loss 1.5677 | label acc 0.420 | val optimal 0.774 | value MAE 2.76 | 7s
step    750 | depth 20 | loss 1.5391 | label acc 0.446 | val optimal 0.791 | value MAE 2.71 | 10s
step   1000 | depth 20 | loss 1.5088 | label acc 0.435 | val optimal 0.805 | value MAE 2.55 | 13s
best validation optimal-move rate 0.805, saved to models\demo.pt
```

### Reading the training output

| Field | Meaning |
| --- | --- |
| `depth` | Current scramble depth. It changes only with `--curriculum-steps`. It is `0` with `--labels exact`, which doesn't scramble. |
| `loss` | Policy loss + `value_weight` x value loss on the current batch. |
| `label acc` | How often the top move matches the label. It is capped well below 100%, because most states have several equally good moves. **Not a quality measure.** |
| `val optimal` | **The metric that matters.** On held-out states, how often the top move really brings the cube one step closer to solved. It is averaged per true distance. |
| `value MAE` | Mean absolute error of the predicted moves left, against the true distance. |

A checkpoint is saved only when `val optimal` beats the best so far, so the
file always holds the best model seen during the run.

The `.csv` log has one row per validation check, with the columns `step`,
`depth`, `loss`, `policy_loss`, `value_loss`, `label_acc`,
`val_optimal_rate`, `val_value_mae`, `lr` and `seconds`. Open it in a
spreadsheet to plot learning curves.

### More training examples

```bash
# Longer self-supervised run
python -m cube_solver.train --steps 20000 --out models/long.pt

# Supervised baseline using the exact table
python -m cube_solver.train --labels exact --steps 10000 --out models/exact.pt

# Piece encoding, half-turn metric, curriculum
python -m cube_solver.train --encoding pieces --half-turns --curriculum-steps 2000 --out models/htm_pieces.pt

# Bigger network
python -m cube_solver.train --width 1024 --n-blocks 6 --steps 20000 --out models/big.pt
```

For reference, after 10,000 steps the scramble-label model reached a
validation optimal-move rate of 0.834 (value MAE 2.45). The exact-label model
reached 0.835 (value MAE 1.60). Self-supervised labels match supervised ones
on move choice. Exact labels give a better value estimate, because the
scramble depth used as the scramble-label value target overestimates the true
distance.

### Resuming a run

```bash
python -m cube_solver.train --resume models/policy.pt --out models/policy.pt --steps 5000 --lr 3e-4
```

- **Taken from the checkpoint:** the network shape, `--encoding` and
  `--half-turns`. Values you pass for these are ignored.
- **Saving:** the run only overwrites the checkpoint once it beats the
  checkpoint's current validation score.
- **Logging:** new rows are appended to the existing `.csv` log, and step
  numbers continue from where the checkpoint stopped.
- **Learning rate:** the schedule restarts, so a lower `--lr` is usually
  better.
- **Always pass `--out`.** Without it, resuming `models/exact.pt` saves to
  `models/policy.pt`.

## Evaluation

```bash
python -m cube_solver.evaluate [flags]
python -m cube_solver.evaluate --help
```

Grades a checkpoint on the frozen held-out states, reporting results for each
true distance. Every run uses the same cubes, so results from different
models can be compared directly.

### Evaluation flags

| Flag | Default | What it does |
| --- | --- | --- |
| `--model` | `models/policy.pt` | Checkpoint to grade. |
| `--split` | `test` | `test` or `val`. Use `test` for final numbers; training already selects on `val`. |
| `--per-distance` | `50` | States per true distance to run each search on. Search is slower than the policy check, so it runs on this subsample. |
| `--methods` | `greedy beam:256` | One or more searches to run, in the formats below. |
| `--oracle` | off | Replace the network with exact answers from the distance table, which shows the best possible score. `--model` must still point to a checkpoint, because the move set is read from it. |
| `--seed` | `0` | Seed for choosing the search subsample. |

### Search methods

| Format | Search | Notes |
| --- | --- | --- |
| `greedy` | Plays the top allowed move that reaches a new state | Fastest, least reliable. Fails when stuck, with no random fallback. |
| `beam:N` | Beam search keeping the N most probable partial solutions | The main method. Larger N is more reliable and slower. Try `beam:16`, `beam:64`, `beam:256` or `beam:1024`. |
| `astar:W` | Batch weighted A*, ranking by `W * moves_so_far + predicted_moves_left` | Uses the value head. Lower W is faster but may give longer solutions. Try `astar:0.6` or `astar:1.0`. |

### Full example

```bash
python -m cube_solver.evaluate --model models/policy.pt --split test --per-distance 20 --methods greedy beam:256 astar:0.8 --seed 0
```

Output:

```
Policy on 1177 test states
dist     n         top1        top2        top3      random value_error
   1     3        1.000       1.000       1.000       0.167       0.185
   2    14        1.000       1.000       1.000       0.190       0.069
   3    60        1.000       1.000       1.000       0.183       0.070
   4   100        1.000       1.000       1.000       0.178       0.129
   5   100        1.000       1.000       1.000       0.192       0.328
   6   100        0.990       1.000       1.000       0.195       1.100
   7   100        0.760       0.900       0.960       0.222       3.623
   8   100        0.570       0.720       0.900       0.222       4.855
   9   100        0.500       0.680       0.790       0.233       4.669
  10   100        0.480       0.640       0.770       0.293       4.269
  11   100        0.540       0.750       0.880       0.512       3.618
  12   100        0.910       1.000       1.000       0.863       2.874
  13   100        1.000       1.000       1.000       0.998       2.003
  14   100        1.000       1.000       1.000       1.000       1.063
all   1177        0.809       0.889       0.941       0.429       2.429

Search: greedy 
dist     n       solved     optimal         gap       nodes          ms
   1     3        1.000       1.000       0.000       1.000      13.262
   2    14        1.000       1.000       0.000       2.000       4.105
   3    20        1.000       1.000       0.000       3.000       9.227
   4    20        1.000       1.000       0.000       4.000       9.245
   5    20        1.000       1.000       0.000       5.000      12.421
   6    20        1.000       0.950       0.200       6.200      17.782
   7    20        0.950       0.750       0.737       8.850      21.582
   8    20        0.800       0.400       5.625      16.900      44.380
   9    20        0.700       0.250       3.000      17.400      46.755
  10    20        0.750       0.050       8.000      21.000      55.128
  11    20        0.500       0.100       7.800      24.400      59.360
  12    20        0.550       0.000       7.455      24.200      66.154
  13    20        0.450       0.000       9.333      26.550      72.778
  14    20        0.600       0.050       8.333      25.400      66.536
all    257        0.790       0.498       3.025      14.354      37.838
solve rate 79.0% (95% CI 73.6% to 83.5%), optimal 49.8%, mean extra moves 3.02, 38 ms per solve

Search: beam {'width': 256}
dist     n       solved     optimal         gap       nodes          ms
   1     3        1.000       1.000       0.000       1.000       3.849
   2    14        1.000       1.000       0.000       7.000       6.357
   3    20        1.000       1.000       0.000      34.000      11.673
   4    20        1.000       1.000       0.000     154.000      12.885
   5    20        1.000       1.000       0.000     410.000      18.481
   6    20        1.000       1.000       0.000     666.000      23.571
   7    20        1.000       1.000       0.000     922.000      27.475
   8    20        1.000       1.000       0.000    1178.000      33.870
   9    20        1.000       1.000       0.000    1434.000      43.583
  10    20        1.000       0.950       0.100    1715.600      46.564
  11    20        1.000       0.900       0.200    1997.200      48.424
  12    20        1.000       1.000       0.000    2202.000      55.732
  13    20        1.000       1.000       0.000    2458.000      56.975
  14    20        1.000       1.000       0.000    2714.000      62.755
all    257        1.000       0.988       0.023    1236.564      34.787
solve rate 100.0% (95% CI 98.5% to 100.0%), optimal 98.8%, mean extra moves 0.02, 35 ms per solve

Search: astar {'weight': 0.8}
dist     n       solved     optimal         gap       nodes          ms
   1     3        1.000       1.000       0.000       1.000       3.007
   2    14        1.000       1.000       0.000       7.000       5.468
   3    20        1.000       1.000       0.000      34.000       7.537
   4    20        1.000       1.000       0.000      98.000      11.143
   5    20        1.000       1.000       0.000     162.000      20.933
   6    20        1.000       1.000       0.000     226.000      20.421
   7    20        1.000       1.000       0.000     290.000      27.029
   8    20        1.000       1.000       0.000     354.000      38.132
   9    20        1.000       1.000       0.000     418.000      43.696
  10    20        1.000       0.900       0.200     494.800      50.659
  11    20        1.000       1.000       0.000     546.000      55.765
  12    20        1.000       1.000       0.000     610.000      58.352
  13    20        1.000       1.000       0.000     674.000      70.003
  14    20        1.000       0.950       0.100     744.400      72.191
all    257        1.000       0.988       0.023     362.354      37.365
solve rate 100.0% (95% CI 98.5% to 100.0%), optimal 98.8%, mean extra moves 0.02, 37 ms per solve
```

### Reading the evaluation output

`dist` is the true optimal distance, and `n` is the number of states in that
bucket. Only 6 states sit at distance 1, so that bucket is small. `all` is the
mean over every state.

**Policy table**, covering every state in the split:

| Column | Meaning |
| --- | --- |
| `top1` | Share of states where the network's first choice is an optimal move. |
| `top2` / `top3` | Share where an optimal move is among the first 2 or 3 choices. The gap between `top1` and `top2` is why beam search works so much better than greedy. |
| `random` | Chance level: the share of moves that are optimal, i.e. what a random move would score. Near distance 14 almost every move is optimal, so high scores there are easy. The middle distances (7 to 11) are the hard part. |
| `value_error` | Mean absolute error of the predicted moves left. |

**Search tables**, covering the `--per-distance` subsample:

| Column | Meaning |
| --- | --- |
| `solved` | Share of states solved. |
| `optimal` | Share solved in exactly the optimal number of moves. Failures count as not optimal. |
| `gap` | Extra moves beyond optimal, for solved states only. `-` means nothing in that bucket was solved. |
| `nodes` | States the network evaluated per solve. |
| `ms` | Time per solve, in milliseconds. |

The summary line gives the solve rate with a 95% Wilson confidence interval,
the optimal rate, the mean extra moves and the mean time per solve.

### More evaluation examples

```bash
# Defaults: greedy and beam:256 on the test split, 50 states per distance
python -m cube_solver.evaluate

# Compare beam widths
python -m cube_solver.evaluate --methods beam:16 beam:64 beam:256 beam:1024

# Grade a different checkpoint
python -m cube_solver.evaluate --model models/exact.pt --methods beam:64 astar:1.0

# Best possible score: exact answers instead of the network
python -m cube_solver.evaluate --oracle --methods greedy

# Tighter confidence intervals
python -m cube_solver.evaluate --per-distance 200 --methods beam:256
```

## Solving cubes from Python

### Solve a random scramble

```python
import numpy as np
from cube_solver import Cube2x2, Solver
from cube_solver.data import scramble_cube

solver = Solver.from_checkpoint("models/policy.pt")   # load once, reuse for many cubes

scramble = scramble_cube(25, np.random.default_rng(7))
cube = Cube2x2().apply_algorithm(scramble)
print("Scramble:", " ".join(scramble))
print(cube)

result = solver.solve(cube, "beam", width=256)
print("Solution:", " ".join(result.moves))
print(f"solved={result.solved} moves={len(result.moves)} nodes={result.nodes} time={result.seconds:.2f}s")
print("check:", cube.copy().apply_algorithm(result.moves).is_solved())
```

Output:

```
Scramble: U' F D' F' R D F U F D' R F R' F' U' F B D' F R' B' L' D D B'
         1  0
         0  3
 5  4    2  1    4  3    4  3
 1  0    3  5    2  1    4  5
         5  0
         2  2
Solution: L B L' D L' L' B' L' D'
solved=True moves=9 nodes=1434 time=0.24s
check: True
```

The first solve is slower because PyTorch is warming up.

### Search options

| Call | Keyword arguments (defaults) |
| --- | --- |
| `solver.solve(cube, "greedy")` | `max_moves=30` |
| `solver.solve(cube, "beam", width=256)` | `width=64`, `max_depth=30` |
| `solver.solve(cube, "astar", weight=0.8)` | `weight=0.8`, `batch_size=64`, `max_nodes=100_000` |

`solve` returns a `SearchResult` with four fields:

| Field | Meaning |
| --- | --- |
| `moves` | The solution, as a list of move strings. |
| `solved` | Whether the search succeeded. |
| `nodes` | States the network evaluated. |
| `seconds` | Time taken. |

`Solver.from_checkpoint(path, device=None)` picks CUDA when it is available.
Pass `device="cpu"` to force the CPU.

### Move notation

- `R` turns the right face clockwise, as seen looking at that face.
- `R'` turns it counter-clockwise.
- `R2` is a half turn.
- The faces are `U` (up), `D` (down), `F` (front), `B` (back), `L` (left) and
  `R` (right).

The network only plays U, R and F moves, with the down-back-left corner held
still. `Solver.solve` translates the solution back to the cube's own
orientation, so solutions can use any face. Quarter-turn models write a half
turn as two quarter turns (for example `L' L'`).

### Working with a cube directly

```python
cube = Cube2x2()                              # solved
cube.apply_move("R").apply_move("U'")         # moves return the cube, so calls chain
cube.apply_algorithm("R U R' U'".split())
cube.is_solved()                              # True or False
clone = cube.copy()                           # independent copy
print(cube)                                   # unfolded net (or cube.print_cube())
```

### Solving your own physical cube

Enter the cube as 24 color numbers, four per face, in the face order U, D, F,
B, L, R.

**Color numbers.** Each number names the face that color belongs to on a
solved cube: `0` = U, `1` = D, `2` = F, `3` = B, `4` = L, `5` = R. With the
standard color scheme, holding white on top and green in front:

| Number | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| Color | white | yellow | green | blue | orange | red |

**Sticker order.** Each face is read top-left, top-right, bottom-left,
bottom-right on this unfolded net:

```
         U U
         U U
 L L   F F   R R   B B
 L L   F F   R R   B B
         D D
         D D
```

- **U face:** read it as if you tilted the cube toward you, so its bottom row
  touches F.
- **D face:** read it as if you tilted the cube away from you, so its top row
  touches F.
- **L, F, R, B faces:** read each one looking straight at it, with U on top.
  B is read from behind the cube.

```python
from cube_solver import Cube2x2, Solver

stickers = [
    0, 0, 0, 0,   # U
    1, 1, 1, 1,   # D
    2, 2, 2, 2,   # F
    3, 3, 3, 3,   # B
    4, 4, 4, 4,   # L
    5, 5, 5, 5,   # R
]
cube = Cube2x2(stickers)
result = Solver.from_checkpoint("models/policy.pt").solve(cube, "beam", width=256)
print(result.solved, " ".join(result.moves))
```

Hold the cube the same way you held it while reading the stickers, then play
the moves in order.

- **Wrong-length input:** anything other than 24 numbers raises `ValueError`.
- **Invalid corner:** an impossible down-back-left corner raises `ValueError`.
- **Other typos may go unnoticed:** a mistake elsewhere can simply fail to
  solve. If `result.solved` is `False`, check what you typed.

## Files the program creates

| File | Created by | Notes |
| --- | --- | --- |
| `models/distances_qtm.npy`, `models/distances_htm.npy` | First run that needs the table (training, evaluation or tests) | Exact distance of every state, 3.7 MB. Safe to delete; it is rebuilt automatically. |
| `models/heldout_qtm.npz`, `models/heldout_htm.npz` | First training or evaluation run | Frozen validation and test states. Deleting it samples a **new** test set, so earlier results are no longer comparable. |
| `models/<name>.pt` | Training | Checkpoint holding the weights plus everything needed to rebuild the model. |
| `models/<name>.csv` | Training | Training log, one row per validation check. |

The distance tables, held-out sets and CSV logs are gitignored.

## How it works

| Module | Role |
| --- | --- |
| `cube_solver/cube.py` | Sticker simulator. Moves are permutation arrays, so a batch of cubes moves at once. Also handles whole-cube rotations, canonicalization and move pruning rules. |
| `cube_solver/state_space.py` | Indexes all 3,674,160 states and runs the breadth-first search for exact distances. |
| `cube_solver/data.py` | Filtered scrambles of 1 to 20 moves, encodings, training batches and frozen held-out states. |
| `cube_solver/model.py` | Residual MLP (~2.3M parameters) with a policy head and a value head. |
| `cube_solver/search.py` | Greedy search, beam search and batch weighted A*. |
| `cube_solver/train.py` / `evaluate.py` | Training loop and evaluation command lines. |

Key design choices:

- **Fixed corner.** The network only sees U, R and F moves, which never move
  the down-back-left corner. Any cube is first rotated into that frame, and the
  solution is translated back to the cube's own orientation.
- **Graded by the exact answer.** Validation tracks the optimal-move rate
  (does the top move really get one step closer?), not label accuracy. Label
  accuracy is capped, because most states have several equally good moves.
- **Search over greedy.** The top move is often wrong, but the right move is
  usually in the top two, and beam search exploits that.

`main.py` is the original single-file version, kept as the baseline. It has
no command-line entry point. Use it from a Python shell with
`from main import *`, then call `demoSolve(Cube2x2(), scrambleLength=5)` or
`solveSuccessRate(100, scrambleLength=10)`.
