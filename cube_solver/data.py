"""Scrambles, input encodings, training batches and the held-out evaluation set.

Everything works on (N, 24) uint8 state arrays so a batch of thousands of
cubes is generated with a handful of NumPy operations.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

from .cube import SOLVED, MoveSet, is_solved_batch
from .state_space import N_STATES, corner_pieces, decode, optimal_move_mask, state_index

ENCODINGS = ("onehot", "pieces")


class Batch(NamedTuple):
    states: np.ndarray         # (N, 24) uint8
    policy_target: np.ndarray  # (N, n_moves) float32 distribution over moves
    value_target: np.ndarray   # (N,) float32 moves left to solve


def random_scrambles(n: int, length: int, move_set: MoveSet, rng: np.random.Generator) -> np.ndarray:
    """(n, length) move indices with redundant sequences (like R R') filtered out."""
    scrambles = np.empty((n, length), dtype=np.intp)
    prev = np.full(n, -1)
    prev2 = np.full(n, -1)
    for t in range(length):
        allowed = move_set.allowed_after(prev, prev2)
        # Random scores with disallowed moves zeroed: the argmax is uniform over allowed moves.
        scrambles[:, t] = np.argmax(rng.random(allowed.shape) * allowed, axis=1)
        prev2, prev = prev, scrambles[:, t].copy()
    return scrambles


def scramble_cube(length: int, rng: np.random.Generator | None = None) -> list[str]:
    """A filtered scramble over all 12 quarter turns, for scrambling a ``Cube2x2``."""
    move_set = MoveSet.all_faces()
    rng = rng or np.random.default_rng()
    return [move_set.names[i] for i in random_scrambles(1, length, move_set, rng)[0]]


def input_dim(encoding: str) -> int:
    return {"onehot": 24 * 6, "pieces": 7 * (7 + 3)}[encoding]


def encode(states: np.ndarray, encoding: str) -> np.ndarray:
    """Network input (N, input_dim) as float32.

    onehot: one binary input per (sticker, color) pair, so no color looks
            "between" two others.
    pieces: per movable corner slot, one-hot of which corner sits there and
            one-hot of its twist. Describes the cube exactly in 70 inputs.
    """
    states = np.atleast_2d(states)
    n = len(states)
    if encoding == "onehot":
        return np.eye(6, dtype=np.float32)[states].reshape(n, -1)
    if encoding == "pieces":
        pieces, twists = corner_pieces(states)
        return np.concatenate([
            np.eye(7, dtype=np.float32)[pieces].reshape(n, -1),
            np.eye(3, dtype=np.float32)[twists].reshape(n, -1),
        ], axis=1)
    raise ValueError(f"unknown encoding {encoding!r}; expected one of {ENCODINGS}")


def scramble_batch(n_scrambles: int, max_depth: int, move_set: MoveSet, rng: np.random.Generator,
                   exclude: np.ndarray | None = None) -> Batch:
    """Self-supervised examples: label each state with the inverse of the last scramble move.

    Every prefix of each scramble is kept, so depths 1..max_depth (and both
    parities) are equally represented. The value target is the scramble depth,
    which is an upper bound on the true distance.
    """
    scrambles = random_scrambles(n_scrambles, max_depth, move_set, rng)
    states = np.tile(SOLVED, (n_scrambles, 1))
    all_states, labels, depths = [], [], []
    for t in range(max_depth):
        states = np.take_along_axis(states, move_set.perms[scrambles[:, t]], axis=1)
        all_states.append(states)
        labels.append(move_set.inverse[scrambles[:, t]])
        depths.append(np.full(n_scrambles, t + 1))
    states, labels, depths = np.concatenate(all_states), np.concatenate(labels), np.concatenate(depths)

    # Never ask the network to turn an already-solved cube, and keep held-out states out.
    keep = ~is_solved_batch(states)
    if exclude is not None:
        keep &= ~exclude[state_index(states)]
    policy = np.eye(len(move_set), dtype=np.float32)[labels[keep]]
    return Batch(states[keep], policy, depths[keep].astype(np.float32))


def exact_batch(n: int, table: np.ndarray, move_set: MoveSet, rng: np.random.Generator,
                exclude: np.ndarray | None = None) -> Batch:
    """Uniformly random states with exact labels from the distance table.

    The policy target spreads probability evenly over every optimal move, so
    the network is never punished for picking an equally good twin move.
    """
    indices = rng.integers(N_STATES, size=n)
    keep = table[indices] > 0
    if exclude is not None:
        keep &= ~exclude[indices]
    indices = indices[keep]
    states = decode(indices)
    optimal = optimal_move_mask(states, table, move_set).astype(np.float32)
    policy = optimal / optimal.sum(axis=1, keepdims=True)
    return Batch(states, policy, table[indices].astype(np.float32))


def load_heldout(table: np.ndarray, move_set: MoveSet, cache_dir: Path = Path("models"),
                 per_distance: int = 100, seed: int = 0) -> dict[str, np.ndarray]:
    """Frozen validation and test state indices, sampled evenly from each true distance.

    Saved to disk so every run is graded on exactly the same cubes.
    """
    path = Path(cache_dir) / f"heldout_{move_set.metric}.npz"
    if path.exists():
        with np.load(path) as saved:
            return {"val": saved["val"], "test": saved["test"]}

    rng = np.random.default_rng(seed)
    val, test = [], []
    for distance in range(1, int(table.max()) + 1):
        bucket = np.flatnonzero(table == distance)
        picks = rng.choice(bucket, size=min(2 * per_distance, len(bucket)), replace=False)
        half = len(picks) // 2
        val.append(picks[:half])
        test.append(picks[half:])
    heldout = {"val": np.concatenate(val), "test": np.concatenate(test)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **heldout)
    return heldout


def training_exclusion(table: np.ndarray, heldout: dict[str, np.ndarray], min_ratio: int = 100) -> np.ndarray:
    """Boolean mask over state indices that training must skip.

    Held-out states are excluded only at distances with at least ``min_ratio``
    times more states than were held out. The shortest distances have just a
    handful of states (6 at distance 1), and holding those out would remove
    them from training entirely.
    """
    held = np.concatenate(list(heldout.values()))
    bucket_sizes = np.bincount(table, minlength=256)
    held_counts = np.bincount(table[held], minlength=256)
    spare = bucket_sizes >= min_ratio * held_counts
    mask = np.zeros(N_STATES, dtype=bool)
    mask[held[spare[table[held]]]] = True
    return mask
