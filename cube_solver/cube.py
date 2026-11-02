"""2x2 cube simulator.

Stickers are stored as 24 color ids (0-5), four per face in the order
U, D, F, B, L, R. Each face is read top-left, top-right, bottom-left,
bottom-right on the unfolded net printed by ``Cube2x2.__str__``.

Every move is a permutation array ``perm`` with ``new[i] = old[perm[i]]``.
Applying a move is ``stickers[perm]``, and the same indexing works on a whole
batch of states at once as ``states[:, perm]``.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

FACES = "UDFBLR"
OPPOSITE = {"U": "D", "D": "U", "F": "B", "B": "F", "L": "R", "R": "L"}

# Clockwise quarter turns, taken verbatim from the original hand-written move
# tables. Primes and half turns are derived from these below.
_CLOCKWISE = {
    "U": [2, 0, 3, 1, 4, 5, 6, 7, 20, 21, 10, 11, 16, 17, 14, 15, 8, 9, 18, 19, 12, 13, 22, 23],
    "D": [0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 18, 19, 12, 13, 22, 23, 16, 17, 14, 15, 20, 21, 10, 11],
    "F": [0, 1, 19, 17, 22, 20, 6, 7, 10, 8, 11, 9, 12, 13, 14, 15, 16, 4, 18, 5, 2, 21, 3, 23],
    "B": [21, 23, 2, 3, 4, 5, 16, 18, 8, 9, 10, 11, 14, 12, 15, 13, 1, 17, 0, 19, 20, 7, 22, 6],
    "L": [15, 1, 13, 3, 8, 5, 10, 7, 0, 9, 2, 11, 12, 6, 14, 4, 18, 16, 19, 17, 20, 21, 22, 23],
    "R": [0, 9, 2, 11, 4, 14, 6, 12, 8, 5, 10, 7, 3, 13, 1, 15, 16, 17, 18, 19, 22, 20, 23, 21],
}

IDENTITY = np.arange(24, dtype=np.intp)
SOLVED = np.repeat(np.arange(6, dtype=np.uint8), 4)


def invert(perm: np.ndarray) -> np.ndarray:
    """Return the permutation that undoes ``perm``."""
    inverse = np.empty_like(perm)
    inverse[perm] = np.arange(len(perm))
    return inverse


def compose(*perms: np.ndarray) -> np.ndarray:
    """Return the single permutation equal to applying ``perms`` left to right."""
    result = IDENTITY
    for perm in perms:
        result = result[perm]
    return result


MOVES: dict[str, np.ndarray] = {}
for _face, _perm in _CLOCKWISE.items():
    _turn = np.array(_perm, dtype=np.intp)
    MOVES[_face] = _turn
    MOVES[_face + "'"] = invert(_turn)
    MOVES[_face + "2"] = _turn[_turn]

QUARTER_TURNS = [face + suffix for face in FACES for suffix in ("", "'")]
HALF_TURNS = [face + "2" for face in FACES]


def inverse_move(move: str) -> str:
    """Return the move that undoes ``move`` (half turns are their own inverse)."""
    if move.endswith("2"):
        return move
    return move[:-1] if move.endswith("'") else move + "'"


def is_solved_batch(states: np.ndarray) -> np.ndarray:
    """True for each state whose faces are all one color.

    Colors are not tied to faces, so a solved cube rotated as a whole counts.
    """
    faces = np.asarray(states).reshape(-1, 6, 4)
    return (faces == faces[:, :, :1]).all(axis=(1, 2))


# --- Whole-cube rotations -------------------------------------------------
# A 2x2 has no fixed centers, so turning two opposite faces in opposite
# directions rotates the whole cube. These three generate all 24 orientations.
def _generate_rotations() -> list[np.ndarray]:
    generators = [compose(MOVES[a], MOVES[b]) for a, b in (("U", "D'"), ("R", "L'"), ("F", "B'"))]
    found = {IDENTITY.tobytes(): IDENTITY}
    frontier = [IDENTITY]
    while frontier:
        next_frontier = []
        for rotation in frontier:
            for generator in generators:
                candidate = compose(rotation, generator)
                if candidate.tobytes() not in found:
                    found[candidate.tobytes()] = candidate
                    next_frontier.append(candidate)
        frontier = next_frontier
    return list(found.values())


ROTATIONS = _generate_rotations()

# Stickers (D, B, L) of the down-back-left corner. The U, R and F moves never
# touch it, so holding it fixed gives the "fixed-corner frame" in which every
# state has exactly one representation.
FIXED_CORNER = np.array([6, 15, 18])


def canonical_rotation(stickers: np.ndarray) -> np.ndarray:
    """Return the whole-cube rotation that puts ``stickers`` in the fixed-corner frame.

    The canonical state is ``stickers[rotation]``.
    """
    for rotation in ROTATIONS:
        if np.array_equal(stickers[rotation][FIXED_CORNER], SOLVED[FIXED_CORNER]):
            return rotation
    raise ValueError("stickers do not describe a reachable 2x2 state")


def translate_move(move: str, rotation: np.ndarray) -> str:
    """Translate a move made in the rotated frame back to the cube's own frame.

    If ``c = s[rotation]`` and move ``m`` is played on ``c``, the equivalent
    move ``m'`` on ``s`` satisfies ``m'[rotation] == rotation[m]``.
    """
    target = rotation[MOVES[move]]
    for candidate in QUARTER_TURNS + HALF_TURNS:
        if np.array_equal(MOVES[candidate][rotation], target):
            return candidate
    raise ValueError(f"no move matches {move!r} under this rotation")


class MoveSet:
    """An ordered list of moves plus the pruning rules for redundant sequences.

    Index ``i`` of a model's output corresponds to ``names[i]``. The rules are
    shared by scramble generation and by search, so neither ever produces
    sequences that cancel themselves out.
    """

    def __init__(self, names: Sequence[str]):
        self.names = tuple(names)
        self.perms = np.stack([MOVES[name] for name in self.names])
        self.index = {name: i for i, name in enumerate(self.names)}
        self.inverse = np.array([self.index[inverse_move(name)] for name in self.names])
        self.has_half_turns = any(name.endswith("2") for name in self.names)

        # forbidden[prev, next] marks pairs that are never useful. The extra
        # last row stands for "no previous move", so prev = -1 indexes it.
        n = len(self.names)
        self._forbidden = np.zeros((n + 1, n), dtype=bool)
        for i, prev in enumerate(self.names):
            for j, nxt in enumerate(self.names):
                undoes = j == self.inverse[i]
                # With half turns available, two turns of one face are one move.
                same_face = self.has_half_turns and prev[0] == nxt[0]
                # Opposite faces commute, so only allow them in one order.
                out_of_order = OPPOSITE[prev[0]] == nxt[0] and FACES.index(nxt[0]) < FACES.index(prev[0])
                self._forbidden[i, j] = undoes or same_face or out_of_order

    @classmethod
    def fixed_corner(cls, half_turns: bool = False) -> "MoveSet":
        """U, R, F moves only: the down-back-left corner never moves."""
        names = ["U", "U'", "R", "R'", "F", "F'"]
        return cls(names + ["U2", "R2", "F2"] if half_turns else names)

    @classmethod
    def all_faces(cls, half_turns: bool = False) -> "MoveSet":
        """All 12 quarter turns (plus 6 half turns), as a person would scramble."""
        return cls(QUARTER_TURNS + HALF_TURNS if half_turns else QUARTER_TURNS)

    @property
    def metric(self) -> str:
        """'htm' when half turns count as one move, else 'qtm'."""
        return "htm" if self.has_half_turns else "qtm"

    def __len__(self) -> int:
        return len(self.names)

    def allowed_after(self, prev: np.ndarray, prev2: np.ndarray) -> np.ndarray:
        """Boolean mask (N, len(self)) of moves allowed after the last two moves.

        ``prev`` and ``prev2`` hold move indices, with -1 meaning "none".
        """
        prev = np.asarray(prev)
        allowed = ~self._forbidden[prev]
        if not self.has_half_turns:
            # Three identical quarter turns equal one turn the other way.
            triple = (prev == np.asarray(prev2)) & (prev >= 0)
            allowed[np.flatnonzero(triple), prev[triple]] = False
        return allowed


class Cube2x2:
    """A single cube. Batched work uses plain (N, 24) arrays instead."""

    def __init__(self, stickers: Iterable[int] | None = None):
        self.stickers = SOLVED.copy() if stickers is None else np.array(stickers, dtype=np.uint8)
        if self.stickers.shape != (24,):
            raise ValueError("a 2x2 cube has exactly 24 stickers")

    def copy(self) -> "Cube2x2":
        return Cube2x2(self.stickers)

    # Equality and hashing use the sticker colors. Do not mutate a cube while
    # it is a dict key or in a set.
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Cube2x2) and np.array_equal(self.stickers, other.stickers)

    def __hash__(self) -> int:
        return hash(self.stickers.tobytes())

    def is_solved(self) -> bool:
        return bool(is_solved_batch(self.stickers)[0])

    def apply_move(self, move: str) -> "Cube2x2":
        self.stickers = self.stickers[MOVES[move]]
        return self

    def apply_algorithm(self, moves: Iterable[str]) -> "Cube2x2":
        for move in moves:
            self.apply_move(move)
        return self

    def __str__(self) -> str:
        """Unfolded net: U on top, then L F R B, then D at the bottom."""
        s = [str(c).rjust(2) for c in self.stickers]
        pad = " " * 8
        return "\n".join([
            pad + " ".join(s[0:2]),
            pad + " ".join(s[2:4]),
            "   ".join(" ".join(s[i:i + 2]) for i in (16, 8, 20, 12)),
            "   ".join(" ".join(s[i:i + 2]) for i in (18, 10, 22, 14)),
            pad + " ".join(s[4:6]),
            pad + " ".join(s[6:8]),
        ])

    def __repr__(self) -> str:
        return f"Cube2x2({self.stickers.tolist()})"

    def print_cube(self) -> None:
        print(self)
