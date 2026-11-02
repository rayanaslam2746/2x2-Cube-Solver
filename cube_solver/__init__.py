"""Self-supervised 2x2 Rubik's Cube solver.

Modules:
    cube         sticker simulator, move permutations, rotations, move sets
    state_space  exact indexing of all 3,674,160 states and the BFS distance table
    data         scrambles, encodings, training batches, held-out states
    model        residual MLP with policy and value heads
    search       greedy, beam search and weighted A* solvers
    train        training loop (python -m cube_solver.train)
    evaluate     grading against the exact table (python -m cube_solver.evaluate)
"""
from .cube import Cube2x2, MoveSet
from .search import Solver

__all__ = ["Cube2x2", "MoveSet", "Solver"]
