import numpy as np
from scoring import score

def test_score_delta0():
    assert 1.0 == score(0, np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]]))
    assert 0.75 == score(0, np.array([[0, 0], [0, 0]]), np.array([[0, 1], [0, 0]]))
    assert 0.0 == score(0, np.array([[0, 0], [0, 0]]), np.array([[1, 1], [1, 1]]))


def test_score_delta1_toad():
    toad_1 = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])
    toad_2 = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])
    assert 1.0 == score(1, toad_1, toad_2)
    assert 1.0 == score(3, toad_1, toad_2)
    assert 30/36 == score(3, np.zeros(toad_2.size).reshape((6,6)), toad_2)


def test_score_block():
    block = np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ])
    block2 = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    assert 1.0 == score(1, block, block)
    assert 1.0 == score(10, block, block)
    assert 10/16 == score(1, block2, block)
    assert 10/16 == score(10, block2, block)