# 2x2-Cube-Solver

A PyTorch 2x2 Rubik's Cube solver that learns with zero human knowledge, in
the style of [EfficientCube (Takano, 2023)](https://arxiv.org/abs/2106.03157).

## Setup

```bash
pip install -r requirements.txt
python -m pytest
```

Run every command from the repository root.

## Training

```bash
python -m cube_solver.train [flags]
```

| Flag | Default | What it does |
| --- | --- | --- |
| `--steps` | `5000` | Number of training steps. |
| `--batch-size` | `1024` | Cubes per step. |
| `--lr` | `0.001` | Peak learning rate. |
| `--weight-decay` | `1e-05` | AdamW weight decay. |
| `--labels` | `scramble` | `scramble` learns from scrambles alone. `exact` learns from the exact optimal moves. |
| `--encoding` | `onehot` | Network input: `onehot` (144 sticker inputs) or `pieces` (70 corner inputs). |
| `--half-turns` | off | Adds U2, R2 and F2 as single moves. |
| `--max-depth` | `0` | Longest scramble. `0` means 20, or 16 with `--half-turns`. |
| `--curriculum-steps` | `0` | Grows scramble depth from 1 to `--max-depth` over this many steps. `0` means off. |
| `--label-smoothing` | `0.0` | Softens the training labels. |
| `--value-weight` | `0.1` | Weight of the moves-left loss. |
| `--stem-width` | `1024` | Width of the first layer. |
| `--width` | `512` | Width of the residual blocks. |
| `--n-blocks` | `3` | Number of residual blocks. |
| `--eval-every` | `250` | Validate and save every N steps. |
| `--seed` | `0` | Random seed. |
| `--out` | `models/policy.pt` | Where to save the model. The log goes next to it as a `.csv`. Overwrites an existing file. |
| `--resume` | *(none)* | Keep training an existing checkpoint. Its network shape, encoding and half-turn setting are reused. |

Example:

```bash
python -m cube_solver.train --steps 10000 --out models/run1.pt
```

Every `--eval-every` steps it prints the loss and `val optimal`: how often the
model's top move is truly optimal. The model is saved whenever `val optimal`
improves.

## Evaluation

```bash
python -m cube_solver.evaluate [flags]
```

| Flag | Default | What it does |
| --- | --- | --- |
| `--model` | `models/policy.pt` | Checkpoint to grade. |
| `--split` | `test` | `test` or `val`. |
| `--per-distance` | `50` | Cubes per solution length to run each search on. |
| `--methods` | `greedy beam:256` | Searches to run (see below). |
| `--oracle` | off | Use the exact answers instead of the model, to show the best possible score. |
| `--seed` | `0` | Seed for picking which cubes are searched. |

| Method | What it does |
| --- | --- |
| `greedy` | Always plays the model's top move. Fastest, least reliable. |
| `beam:N` | Keeps the N best partial solutions. Larger N is more reliable but slower. |
| `astar:W` | Weighted A* guided by the model's moves-left estimate. Lower W is faster but can give longer solutions. |

Example:

```bash
python -m cube_solver.evaluate --model models/run1.pt --methods greedy beam:256 astar:0.8
```

It prints results for each solution length, then a summary line per method:
solve rate, share of solutions that were optimal, average extra moves and
time per solve.
