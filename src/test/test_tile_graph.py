import numpy as np
import pytest
from tqdm import tqdm

from tile_graph import DynamicProg, DFS, TileGraph
from simulator import life_step
from bitmap import generate_all
from scoring import score

tile_graph = TileGraph()

@pytest.mark.parametrize("alg_class", [DynamicProg, DFS])
def test_step_back_simple(alg_class):
    alg = alg_class(tile_graph)

    block = np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ])
    A = alg.step_back(block)
    B = life_step(A)
    assert (B == block).all()

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
    A = alg.step_back(toad_2)
    assert (A == toad_1).all()


# DynamicProg cannot unfortunately pass this test :(
@pytest.mark.parametrize("alg_class", [DFS])
def test_step_back_all_3x3(alg_class):
    alg = alg_class(tile_graph)

    for Aorig in tqdm(generate_all(3,3)):
        X = np.copy(Aorig)
        X = life_step(X)
        A = alg.step_back(X)

        sc = score(1, X, A)
        assert sc == 1.0
