"""Solvers that turn a move policy (and value estimate) into a full solution.

Each search takes an ``Evaluator``: a function from (N, 24) states to
``(log_probs (N, n_moves), values (N,))``. That can be the trained network or,
for testing, an oracle built from the exact distance table.
"""
from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .cube import Cube2x2, MoveSet, canonical_rotation, is_solved_batch, translate_move
from .data import encode
from .model import PolicyValueNet, load_checkpoint

Evaluator = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass
class SearchResult:
    moves: list[str]
    solved: bool
    nodes: int      # states evaluated by the network
    seconds: float


def network_evaluator(model: PolicyValueNet, encoding: str, device: str | torch.device,
                      batch_size: int = 8192) -> Evaluator:
    """Wrap a loaded model (in eval mode) as an ``Evaluator``."""
    model.eval()

    @torch.no_grad()
    def evaluate(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        log_probs, values = [], []
        for start in range(0, len(states), batch_size):
            x = torch.from_numpy(encode(states[start:start + batch_size], encoding)).to(device)
            logits, value = model(x)
            log_probs.append(F.log_softmax(logits, dim=-1).cpu().numpy())
            values.append(value.cpu().numpy())
        return np.concatenate(log_probs), np.concatenate(values)

    return evaluate


def _result(path, move_set: MoveSet, solved: bool, nodes: int, start: float) -> SearchResult:
    return SearchResult([move_set.names[i] for i in path], solved, nodes, time.perf_counter() - start)


def greedy(state: np.ndarray, evaluate: Evaluator, move_set: MoveSet, max_moves: int = 30) -> SearchResult:
    """Play the most likely move each step, skipping redundant moves and revisited states.

    There is no random fallback: if every move leads somewhere already seen,
    the solve is reported as a failure.
    """
    start = time.perf_counter()
    path: list[int] = []
    seen = {state.tobytes()}
    for _ in range(max_moves):
        if is_solved_batch(state)[0]:
            break
        log_probs, _ = evaluate(state[None])
        prev = path[-1] if path else -1
        prev2 = path[-2] if len(path) > 1 else -1
        allowed = move_set.allowed_after(np.array([prev]), np.array([prev2]))[0]

        # Take the best allowed move that reaches a new state.
        for move in np.argsort(-log_probs[0]):
            child = state[move_set.perms[move]]
            if allowed[move] and child.tobytes() not in seen:
                break
        else:
            break
        path.append(int(move))
        seen.add(child.tobytes())
        state = child
    return _result(path, move_set, bool(is_solved_batch(state)[0]), len(path), start)


def beam_search(state: np.ndarray, evaluate: Evaluator, move_set: MoveSet, width: int = 64,
                max_depth: int = 30) -> SearchResult:
    """Keep the ``width`` most probable partial solutions, scored by summed log-probability.

    This is the EfficientCube search: even when the top move is wrong, the
    right one is usually in the top few, and the beam keeps it alive.
    """
    start = time.perf_counter()
    if is_solved_batch(state)[0]:
        return _result([], move_set, True, 0, start)

    n_moves = len(move_set)
    states = state[None]
    paths = np.zeros((1, 0), dtype=np.intp)
    scores = np.zeros(1)
    nodes = 0
    for _ in range(max_depth):
        log_probs, _ = evaluate(states)
        nodes += len(states)

        # Score every child; redundant moves (like R after R') get -inf.
        none = np.full(len(states), -1)
        prev = paths[:, -1] if paths.shape[1] > 0 else none
        prev2 = paths[:, -2] if paths.shape[1] > 1 else none
        candidate_scores = scores[:, None] + log_probs
        candidate_scores[~move_set.allowed_after(prev, prev2)] = -np.inf
        candidate_scores = candidate_scores.ravel()
        children = states[:, move_set.perms].reshape(-1, 24)

        # Walk candidates best-first, so the first copy of a duplicate state is its best path.
        order = np.argsort(-candidate_scores, kind="stable")
        order = order[np.isfinite(candidate_scores[order])]
        _, first = np.unique(children[order], axis=0, return_index=True)
        order = order[np.sort(first)]

        parent, move = np.divmod(order, n_moves)
        solved = is_solved_batch(children[order])
        if solved.any():
            best = np.flatnonzero(solved)[0]
            return _result([*paths[parent[best]], move[best]], move_set, True, nodes, start)

        order, parent, move = order[:width], parent[:width], move[:width]
        states = children[order]
        paths = np.column_stack([paths[parent], move])
        scores = candidate_scores[order]
    return _result([], move_set, False, nodes, start)


def weighted_astar(state: np.ndarray, evaluate: Evaluator, move_set: MoveSet, weight: float = 0.8,
                   batch_size: int = 64, max_nodes: int = 100_000) -> SearchResult:
    """Batch weighted A* (DeepCubeA style), guided by the value head.

    Nodes are ranked by ``f = weight * g + h``, where g is moves made so far
    and h is the predicted moves left. A weight below 1 trades optimality for
    speed. Each iteration expands the ``batch_size`` best nodes at once so
    the network is called on batches rather than single states.
    """
    start = time.perf_counter()
    if is_solved_batch(state)[0]:
        return _result([], move_set, True, 0, start)

    root = state.tobytes()
    best_g = {root: 0}
    parent: dict[bytes, tuple[bytes | None, int]] = {root: (None, -1)}
    tiebreak = itertools.count()
    _, h = evaluate(state[None])
    open_heap = [(max(float(h[0]), 0.0), next(tiebreak), 0, root)]

    def path_to(key: bytes) -> list[int]:
        path = []
        while parent[key][0] is not None:
            key, move = parent[key]
            path.append(move)
        return path[::-1]

    nodes = 0
    while open_heap and nodes < max_nodes:
        # Pop the best batch, skipping stale entries that a shorter path replaced.
        batch = []
        while open_heap and len(batch) < batch_size:
            _, _, g, key = heapq.heappop(open_heap)
            if g == best_g[key]:
                batch.append(key)
        if not batch:
            break
        nodes += len(batch)

        parents = np.stack([np.frombuffer(key, dtype=np.uint8) for key in batch])
        prev = np.array([parent[key][1] for key in batch])
        prev2 = np.array([parent[parent[key][0]][1] if parent[key][0] is not None else -1 for key in batch])
        allowed = move_set.allowed_after(prev, prev2)
        children = parents[:, move_set.perms]

        new_keys, new_states = [], []
        for b, key in enumerate(batch):
            g = best_g[key] + 1
            for move in np.flatnonzero(allowed[b]):
                child = children[b, move]
                child_key = child.tobytes()
                if g < best_g.get(child_key, np.inf):
                    best_g[child_key] = g
                    parent[child_key] = (key, int(move))
                    if is_solved_batch(child)[0]:
                        return _result(path_to(child_key), move_set, True, nodes, start)
                    new_keys.append(child_key)
                    new_states.append(child)

        if new_keys:
            _, h = evaluate(np.stack(new_states))
            for child_key, child_h in zip(new_keys, h):
                g = best_g[child_key]
                heapq.heappush(open_heap, (weight * g + max(float(child_h), 0.0), next(tiebreak), g, child_key))
    return _result([], move_set, False, nodes, start)


SEARCHES = {"greedy": greedy, "beam": beam_search, "astar": weighted_astar}


class Solver:
    """Solve any ``Cube2x2``, in any orientation, with a trained network."""

    def __init__(self, evaluate: Evaluator, move_set: MoveSet):
        self.evaluate = evaluate
        self.move_set = move_set

    @classmethod
    def from_checkpoint(cls, path: Path, device: str | None = None) -> "Solver":
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, config = load_checkpoint(path, device)
        evaluate = network_evaluator(model, config["encoding"], device)
        return cls(evaluate, MoveSet.fixed_corner(config["half_turns"]))

    def solve(self, cube: Cube2x2, method: str = "beam", **kwargs) -> SearchResult:
        """Search in the fixed-corner frame, then translate the moves back to the cube's own frame."""
        rotation = canonical_rotation(cube.stickers)
        result = SEARCHES[method](cube.stickers[rotation], self.evaluate, self.move_set, **kwargs)
        result.moves = [translate_move(move, rotation) for move in result.moves]
        return result
