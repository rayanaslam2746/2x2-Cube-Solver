"""The exact state space of the 2x2 in the fixed-corner frame.

With the down-back-left corner held in place, a state is fully described by
which of the other 7 corners sits in each slot (7! ways) and how 6 of them are
twisted (3^6 ways; the 7th twist is forced). That gives a dense index for all
3,674,160 states, so a breadth-first search can store the exact optimal
distance of every state in a 3.7 MB array.
"""
from __future__ import annotations

from math import factorial
from pathlib import Path

import numpy as np

from .cube import SOLVED, MoveSet

# Sticker positions of the 8 corner slots. Each triple starts with the U or D
# sticker and continues clockwise around the corner, so twisting a corner is
# a cyclic shift of its triple. The fixed DBL corner is last.
CORNER_SLOTS = np.array([
    [3, 20, 9],    # URF
    [2, 8, 17],    # UFL
    [0, 16, 13],   # ULB
    [1, 12, 21],   # UBR
    [5, 11, 22],   # DFR
    [4, 19, 10],   # DLF
    [7, 23, 14],   # DRB
    [6, 15, 18],   # DBL (fixed)
])
N_MOVABLE = 7
N_STATES = factorial(N_MOVABLE) * 3 ** (N_MOVABLE - 1)  # 3,674,160
UNREACHED = np.uint8(255)

# Colors of each corner piece as seen in its home slot.
PIECE_COLORS = SOLVED[CORNER_SLOTS]

_FACTORIALS = np.array([factorial(N_MOVABLE - 1 - i) for i in range(N_MOVABLE)])
_TWIST_WEIGHTS = 3 ** np.arange(N_MOVABLE - 2, -1, -1)
_LATER = np.triu(np.ones((N_MOVABLE, N_MOVABLE), dtype=bool), k=1)


def _color_code(triples: np.ndarray) -> np.ndarray:
    triples = triples.astype(np.intp)
    return (triples[..., 0] * 6 + triples[..., 1]) * 6 + triples[..., 2]


# Lookup from a slot's three colors to (piece, twist). Twist t means the slot
# shows the piece's colors shifted by t; mirrored triples stay at -1 because
# no legal move can produce them.
_PIECE_OF = np.full(216, -1, dtype=np.intp)
_TWIST_OF = np.full(216, -1, dtype=np.intp)
for _piece, _colors in enumerate(PIECE_COLORS[:N_MOVABLE]):
    for _twist in range(3):
        _code = _color_code(np.roll(_colors, -_twist))
        _PIECE_OF[_code], _TWIST_OF[_code] = _piece, _twist


def corner_pieces(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(pieces, twists)``, each (N, 7): the corner in each movable slot and its twist."""
    states = np.atleast_2d(states)
    codes = _color_code(states[:, CORNER_SLOTS[:N_MOVABLE]])
    pieces, twists = _PIECE_OF[codes], _TWIST_OF[codes]
    if (pieces < 0).any():
        raise ValueError("state is not in the fixed-corner frame; canonicalize it first")
    return pieces, twists


def state_index(states: np.ndarray) -> np.ndarray:
    """Map fixed-corner states (N, 24) to unique indices in ``[0, N_STATES)``."""
    pieces, twists = corner_pieces(states)
    # Lehmer code: for each slot, count the later slots holding a smaller piece.
    lehmer = ((pieces[:, None, :] < pieces[:, :, None]) & _LATER).sum(axis=2)
    perm_index = lehmer @ _FACTORIALS
    twist_index = twists[:, :-1] @ _TWIST_WEIGHTS
    return perm_index * 3 ** (N_MOVABLE - 1) + twist_index


def decode(indices: np.ndarray) -> np.ndarray:
    """Inverse of ``state_index``: rebuild (N, 24) sticker arrays."""
    indices = np.asarray(indices, dtype=np.int64).ravel()
    perm_index, twist_index = np.divmod(indices, 3 ** (N_MOVABLE - 1))
    n = len(indices)

    # Undo the Lehmer code: slot i holds the (rank + 1)-th piece not yet used.
    pieces = np.empty((n, N_MOVABLE), dtype=np.intp)
    used = np.zeros((n, N_MOVABLE), dtype=bool)
    for i in range(N_MOVABLE):
        rank = (perm_index // _FACTORIALS[i]) % (N_MOVABLE - i)
        unused_so_far = np.cumsum(~used, axis=1)
        pieces[:, i] = np.argmax((unused_so_far == rank[:, None] + 1) & ~used, axis=1)
        used[np.arange(n), pieces[:, i]] = True

    # The last twist is whatever makes the total a multiple of 3.
    twists = np.empty((n, N_MOVABLE), dtype=np.intp)
    twists[:, :-1] = (twist_index[:, None] // _TWIST_WEIGHTS) % 3
    twists[:, -1] = -twists[:, :-1].sum(axis=1) % 3

    states = np.tile(SOLVED, (n, 1))
    for i in range(N_MOVABLE):
        shift = (np.arange(3) + twists[:, i, None]) % 3
        states[:, CORNER_SLOTS[i]] = np.take_along_axis(PIECE_COLORS[pieces[:, i]], shift, axis=1)
    return states


def build_distance_table(move_set: MoveSet, chunk: int = 250_000) -> np.ndarray:
    """Breadth-first search from solved: optimal move count for every state index."""
    table = np.full(N_STATES, UNREACHED, dtype=np.uint8)
    table[state_index(SOLVED)] = 0
    frontier, depth = SOLVED[None].copy(), 0
    while len(frontier):
        layer = []
        # Chunking keeps memory flat at the widest layers (~1.3M states).
        for start in range(0, len(frontier), chunk):
            children = frontier[start:start + chunk][:, move_set.perms].reshape(-1, 24)
            indices = state_index(children)
            fresh = table[indices] == UNREACHED
            indices, first = np.unique(indices[fresh], return_index=True)
            table[indices] = depth + 1
            layer.append(children[fresh][first])
        frontier, depth = np.concatenate(layer), depth + 1
    return table


def load_distance_table(move_set: MoveSet, cache_dir: Path = Path("models")) -> np.ndarray:
    """Load the cached distance table for this move metric, building it on first use."""
    path = Path(cache_dir) / f"distances_{move_set.metric}.npy"
    if path.exists():
        return np.load(path)
    table = build_distance_table(move_set)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, table)
    return table


def optimal_move_mask(states: np.ndarray, table: np.ndarray, move_set: MoveSet) -> np.ndarray:
    """(N, len(move_set)) mask of the moves that bring each state one step closer to solved."""
    states = np.atleast_2d(states)
    distance = table[state_index(states)].astype(np.int16)
    children = states[:, move_set.perms].reshape(-1, 24)
    child_distance = table[state_index(children)].reshape(len(states), len(move_set))
    return child_distance == distance[:, None] - 1
