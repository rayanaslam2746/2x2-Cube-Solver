"""Grade a trained model against the exact distance table.

Usage:
    python -m cube_solver.evaluate --model models/policy.pt --methods greedy beam:16 beam:256 astar:0.8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .cube import MoveSet
from .data import load_heldout
from .search import SEARCHES, Evaluator, network_evaluator
from .model import load_checkpoint
from .state_space import decode, load_distance_table, optimal_move_mask, state_index


def oracle_evaluator(table: np.ndarray, move_set: MoveSet) -> Evaluator:
    """A perfect "network" read straight from the table: the upper bound for any model."""
    def evaluate(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        optimal = optimal_move_mask(states, table, move_set)
        probs = optimal / np.maximum(optimal.sum(axis=1, keepdims=True), 1)
        with np.errstate(divide="ignore"):
            return np.log(probs), table[state_index(states)].astype(np.float32)
    return evaluate


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a rate, well behaved near 0% and 100%."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - half, center + half


def policy_report(evaluate: Evaluator, states: np.ndarray, distances: np.ndarray,
                  table: np.ndarray, move_set: MoveSet) -> dict[str, np.ndarray]:
    """Per-state policy metrics: is an optimal move in the top 1, 2 or 3? Plus value error."""
    log_probs, values = evaluate(states)
    optimal = optimal_move_mask(states, table, move_set)
    ranked = np.argsort(-log_probs, axis=1)
    report = {f"top{k}": np.take_along_axis(optimal, ranked[:, :k], axis=1).any(axis=1) for k in (1, 2, 3)}
    # Chance level: the share of moves that are optimal, i.e. what a random move scores.
    report["random"] = optimal.mean(axis=1)
    report["value_error"] = np.abs(values - distances)
    return report


def search_report(search, evaluate: Evaluator, states: np.ndarray, distances: np.ndarray,
                  move_set: MoveSet, **kwargs) -> dict[str, np.ndarray]:
    """Run one search on every state and collect per-state outcomes."""
    results = [search(state, evaluate, move_set, **kwargs) for state in states]
    solved = np.array([r.solved for r in results])
    lengths = np.array([len(r.moves) for r in results])
    return {
        "solved": solved,
        # Failures count as non-optimal, so this rate is over every attempt.
        "optimal": solved & (lengths == distances),
        "gap": np.where(solved, lengths - distances, np.nan),
        "nodes": np.array([r.nodes for r in results]),
        "ms": np.array([r.seconds * 1000 for r in results]),
    }


def print_by_distance(title: str, distances: np.ndarray, columns: dict[str, np.ndarray]) -> None:
    """Print the mean of each column for every true distance, then overall."""
    columns = {name: values.astype(float) for name, values in columns.items()}
    print(f"\n{title}")
    print("dist     n " + "".join(f"{name:>12}" for name in columns))
    for d in [*np.unique(distances), None]:
        rows = np.ones_like(distances, dtype=bool) if d is None else distances == d
        label = "all " if d is None else f"{int(d):>4}"
        # A column can be all NaN in a bucket (e.g. extra moves when nothing was solved).
        cells = "".join(f"{np.nanmean(values[rows]):>12.3f}" if not np.isnan(values[rows]).all()
                        else f"{'-':>12}" for values in columns.values())
        print(f"{label} {rows.sum():>5} {cells}")


def parse_method(spec: str) -> tuple[str, dict]:
    """'greedy', 'beam:256' (width) or 'astar:0.8' (weight)."""
    name, _, arg = spec.partition(":")
    if name not in SEARCHES:
        raise argparse.ArgumentTypeError(f"unknown method {name!r}")
    if not arg:
        return name, {}
    return name, {"width": int(arg)} if name == "beam" else {"weight": float(arg)} if name == "astar" else {}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=Path("models/policy.pt"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--per-distance", type=int, default=50, help="states per true distance to search")
    parser.add_argument("--methods", nargs="+", type=parse_method, default=[parse_method("greedy"), parse_method("beam:256")])
    parser.add_argument("--oracle", action="store_true", help="use the exact table instead of the model")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = load_checkpoint(args.model, device)
    move_set = MoveSet.fixed_corner(config["half_turns"])
    table = load_distance_table(move_set)
    evaluate = oracle_evaluator(table, move_set) if args.oracle else network_evaluator(model, config["encoding"], device)

    indices = load_heldout(table, move_set)[args.split]
    states, distances = decode(indices), table[indices].astype(int)
    policy = policy_report(evaluate, states, distances, table, move_set)
    print_by_distance(f"Policy on {len(states)} {args.split} states", distances, policy)

    # Search is slower, so it runs on a fixed subsample per distance.
    rng = np.random.default_rng(args.seed)
    picks = np.concatenate([rng.permutation(np.flatnonzero(distances == d))[:args.per_distance]
                            for d in np.unique(distances)])
    for name, kwargs in args.methods:
        report = search_report(SEARCHES[name], evaluate, states[picks], distances[picks], move_set, **kwargs)
        print_by_distance(f"Search: {name} {kwargs or ''}", distances[picks], report)
        n, solved, optimal = len(picks), report["solved"].sum(), report["optimal"].sum()
        lo, hi = wilson_interval(solved, n)
        print(f"solve rate {solved / n:.1%} (95% CI {lo:.1%} to {hi:.1%}), "
              f"optimal {optimal / n:.1%}, mean extra moves {np.nanmean(report['gap']):.2f}, "
              f"{np.mean(report['ms']):.0f} ms per solve")


if __name__ == "__main__":
    main()
