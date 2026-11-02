import numpy as np
import pytest

from cube_solver.cube import (
    HALF_TURNS, MOVES, QUARTER_TURNS, Cube2x2, MoveSet, canonical_rotation, is_solved_batch,
)
from cube_solver.data import scramble_batch, scramble_cube
from cube_solver.evaluate import oracle_evaluator
from cube_solver.search import Solver
from cube_solver.state_space import (
    CORNER_SLOTS, N_STATES, build_distance_table, decode, optimal_move_mask, state_index,
)

# Published number of 2x2 states at each optimal distance.
QTM_COUNTS = [1, 6, 27, 120, 534, 2256, 8969, 33058, 114149, 360508, 930588, 1350852, 782536, 90280, 276]
HTM_COUNTS = [1, 9, 54, 321, 1847, 9992, 50136, 227536, 870072, 1887748, 623800, 2644]


@pytest.fixture(scope="session")
def qtm_table():
    return build_distance_table(MoveSet.fixed_corner())


def test_quarter_turn_distance_distribution(qtm_table):
    assert np.bincount(qtm_table).tolist() == QTM_COUNTS
    assert sum(QTM_COUNTS) == N_STATES


def test_half_turn_distance_distribution():
    table = build_distance_table(MoveSet.fixed_corner(half_turns=True))
    assert np.bincount(table).tolist() == HTM_COUNTS


@pytest.mark.parametrize("move", QUARTER_TURNS + HALF_TURNS)
def test_corner_slots_move_as_units(move):
    # Each slot's stickers must come from one slot, in the same cyclic order.
    slots = {tuple(np.roll(slot, k)) for slot in CORNER_SLOTS for k in range(3)}
    for slot in CORNER_SLOTS:
        assert tuple(MOVES[move][slot]) in slots


def test_index_round_trip():
    indices = np.random.default_rng(0).integers(N_STATES, size=20_000)
    assert np.array_equal(state_index(decode(indices)), indices)


def test_every_unsolved_state_has_an_optimal_move(qtm_table):
    moves = MoveSet.fixed_corner()
    indices = np.random.default_rng(1).integers(N_STATES, size=5_000)
    indices = indices[qtm_table[indices] > 0]
    assert optimal_move_mask(decode(indices), qtm_table, moves).any(axis=1).all()


def test_scramble_batch_labels_undo_the_last_move(qtm_table):
    moves = MoveSet.fixed_corner()
    batch = scramble_batch(200, 14, moves, np.random.default_rng(2))
    assert not is_solved_batch(batch.states).any()
    # The label undoes the last scramble move, landing on the scramble one move shorter.
    labels = batch.policy_target.argmax(axis=1)
    undone = np.take_along_axis(batch.states, moves.perms[labels], axis=1)
    assert (qtm_table[state_index(undone)] <= batch.value_target - 1).all()


@pytest.mark.parametrize("method, kwargs", [("greedy", {}), ("beam", {"width": 1}), ("astar", {})])
def test_oracle_solves_any_orientation_optimally(qtm_table, method, kwargs):
    # Scramble with all 12 moves, so the solver must canonicalize and translate back.
    moves = MoveSet.fixed_corner()
    solver = Solver(oracle_evaluator(qtm_table, moves), moves)
    rng = np.random.default_rng(3)
    for _ in range(10):
        cube = Cube2x2().apply_algorithm(scramble_cube(25, rng))
        result = solver.solve(cube, method, **kwargs)
        assert result.solved
        assert cube.copy().apply_algorithm(result.moves).is_solved()
        if method != "astar":   # weighted A* trades optimality for speed
            canonical = cube.stickers[canonical_rotation(cube.stickers)]
            assert len(result.moves) == qtm_table[state_index(canonical)][0]
