import numpy as np
import pytest

from cube_solver.cube import (
    HALF_TURNS, IDENTITY, MOVES, QUARTER_TURNS, ROTATIONS, SOLVED, Cube2x2, MoveSet,
    compose, inverse_move, is_solved_batch,
)

ALL_MOVES = QUARTER_TURNS + HALF_TURNS


@pytest.mark.parametrize("move", ALL_MOVES)
def test_every_move_is_a_permutation(move):
    assert sorted(MOVES[move]) == list(range(24))


@pytest.mark.parametrize("move", QUARTER_TURNS)
def test_move_then_inverse_and_four_turns_are_identity(move):
    assert np.array_equal(compose(MOVES[move], MOVES[inverse_move(move)]), IDENTITY)
    assert np.array_equal(compose(*[MOVES[move]] * 4), IDENTITY)
    assert np.array_equal(compose(MOVES[move], MOVES[move]), MOVES[move[0] + "2"])


@pytest.mark.parametrize("move", QUARTER_TURNS)
def test_quarter_turn_moves_exactly_12_stickers(move):
    assert (MOVES[move] != IDENTITY).sum() == 12


@pytest.mark.parametrize("a, b", [("U", "D"), ("F", "B"), ("L", "R")])
def test_opposite_faces_commute(a, b):
    assert np.array_equal(compose(MOVES[a], MOVES[b]), compose(MOVES[b], MOVES[a]))


def test_rotations_keep_cube_solved_and_give_24_orientations():
    assert len(ROTATIONS) == 24
    assert len({r.tobytes() for r in ROTATIONS}) == 24
    assert is_solved_batch(np.stack([SOLVED[r] for r in ROTATIONS])).all()


def test_cube_api():
    cube = Cube2x2()
    clone = cube.copy()
    assert cube.apply_move("R") is cube           # moves return the cube for chaining
    assert not cube.is_solved() and clone.is_solved()
    assert cube != clone and len({cube, clone, clone.copy()}) == 2
    cube.apply_algorithm(["R'"])
    assert cube == clone


def test_move_set_pruning():
    moves = MoveSet.fixed_corner()
    r, r_prime = moves.index["R"], moves.index["R'"]
    allowed = moves.allowed_after(np.array([r, r, -1]), np.array([-1, r, -1]))
    assert not allowed[0, r_prime]                 # no immediate inverse
    assert allowed[1].sum() == len(moves) - 2      # after R R: neither R' nor a third R
    assert allowed[2].all()                        # anything goes on the first move

    half = MoveSet.fixed_corner(half_turns=True)
    blocked = ~half.allowed_after(np.array([half.index["U"]]), np.array([-1]))[0]
    assert {half.names[i] for i in np.flatnonzero(blocked)} == {"U", "U'", "U2"}
