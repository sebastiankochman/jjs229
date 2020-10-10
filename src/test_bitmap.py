from bitmap import generate_bitmaps


def test_generate_bitmaps():
    Xs = list(generate_bitmaps(2, 2))
    assert 16 == len(Xs)
    assert (Xs[0] == [[0,0],[0,0]]).all()
    assert (Xs[1] == [[1,0],[0,0]]).all()
    assert (Xs[2] == [[0,1],[0,0]]).all()
    assert (Xs[3] == [[1,1],[0,0]]).all()
    assert (Xs[4] == [[0,0],[1,0]]).all()
