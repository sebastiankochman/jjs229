from bitmap import generate_all, generate_train_set
from simulator import life_step


def test_generate_bitmaps():
    Xs = list(generate_all(2, 2))
    assert 16 == len(Xs)
    assert (Xs[0] == [[0,0],[0,0]]).all()
    assert (Xs[1] == [[1,0],[0,0]]).all()
    assert (Xs[2] == [[0,1],[0,0]]).all()
    assert (Xs[3] == [[1,1],[0,0]]).all()
    assert (Xs[4] == [[0,0],[1,0]]).all()


def test_generate_train_set():
    xs = list(generate_train_set(10, 234))
    assert len(xs) == 10
    for delta, start, stop in xs:
        for i in range(delta):
            start = life_step(start)
        assert (start == stop).all()

