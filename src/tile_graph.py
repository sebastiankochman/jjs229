# Algorithm based on dynamic programming

import numpy as np
import time
from simulator import life_step
from bitmap import generate_all, generate_inf_cases
from scoring import score
from tqdm import tqdm

class TileGraph:
    def __init__(self):
        self.tiles, self.prev, self.horiz, self.verti = self.preprocess()

    @staticmethod
    def preprocess():
        # Tiles - all possible titles 3x3
        T = list(generate_all(3, 3))
        assert (len(T) == 512)

        # Backward possibilities
        # dict: central bit -> list of possible prev tiles 3x3
        B = [[], []]
        for i, tile in enumerate(T):
            next_tile = life_step(tile)
            B[int(next_tile[1][1])].append(i)

        # Fun fact:
        # >>> len(B[0])
        # 372
        # >>> len(B[1])
        # 140

        # Matrix horiz[i,j] - true if tile j can be put horizontally on the right side of tile i.
        # 0 0 1     0 1 0    0 0 1 0
        # 0[1]0  +  1[0]0 =  0[1|0]0
        # 0 0 0     0 0 1    0 0 0 1
        horiz = np.zeros((len(T), len(T)), dtype=np.bool)

        # Matrix verti[i,j] - true if tile j can be put vertically under tile i.
        verti = np.zeros((len(T), len(T)), dtype=np.bool)

        # Diagonal relationships -- they don't seem to be needed.
        #diago_se = np.zeros((len(T), len(T)), dtype=np.bool)
        #diago_sw = np.zeros((len(T), len(T)), dtype=np.bool)
        for i, x in enumerate(T):
            for j, y in enumerate(T):
                # Check if left tile is compatible with the right tile.
                # I.e., left tile's right side is equal to right tile's left side.
                horiz[i, j] = (x[:, (1, 2)] == y[:, (0, 1)]).all()

                # Check if upper tile is compatible with the lower tile.
                # I.e., upper tile's lower side is equal to lower tile's upper side.
                verti[i, j] = (x[(1, 2), :] == y[(0, 1), :]).all()

                # Diagonal relationships -- they don't seem to be needed.
                #diago_se[i, j] = (x[(1, 2), (1, 2)] == y[(0, 1), (0, 1)]).all()
                #diago_sw[i, j] = (x[(1, 2), (0, 1)] == y[(0, 1), (1, 2)]).all()

        return T, B, horiz, verti

    @staticmethod
    def get_compatible_Y(X, Y, C):
        """
        :param X: first set of tile ids
        :param Y: second set of tile ids
        :param C: compatibility matrix (R or D)
        :return:
        """
        return [y for y in Y if any(C[x][y] for x in X)]

    @staticmethod
    def get_compatible_X(X, Y, C):
        """
        :param X: first set of tile ids
        :param Y: second set of tile ids
        :param C: compatibility matrix (R or D)
        :return:
        """
        return [x for x in X if any(C[x][y] for y in Y)]


class DFS:
    """
    Depth-First Search - examines all possible paths of compatible tiles until finding the solution.
    Slow but reliable.
    """
    def __init__(self, tile_graph=None):
        self.G = tile_graph if tile_graph is not None else TileGraph()

    def step_back(self, F, verbose=False):
        """
        :param F: final bitmap (boolean matrix)
        :return:
        """
        tile_board = np.full_like(F, -1, dtype=np.int)
        r = self.dfs(F, tile_board, 0, 0, verbose)
        if r is None:
            raise Exception('No previous state found.')

        # Set central bit of the tile to the result bitmap.
        A = np.array([[self.G.tiles[tile_board[i,j]][1][1] for j in range(tile_board.shape[1])] for i in range(tile_board.shape[0])])
        return A

    def dfs(self, F, tile_board, i, j, verbose=False):
        """
        :param tile_board: matrix m x n with id of a tile chosen on each pixel or -1 if not chosen yet
        :param i: current row
        :param j: current column
        :return: finished tile_board or None if couldn't find the path
        """
        m = len(F)
        n = len(F[0])

        if tile_board[i][j] != -1:
            # Success!
            # We have been here already, which means we have iterated over the whole board and found the solution.
            assert i == 0 and j == 0
            return tile_board

        possible_tiles = self.G.prev[F[i][j]].copy()
        if tile_board[i-1][j] != -1:
            possible_tiles = self.G.get_compatible_Y([tile_board[i-1][j]], possible_tiles, self.G.verti)
        if tile_board[i][j-1] != -1:
            possible_tiles = self.G.get_compatible_Y([tile_board[i][j-1]], possible_tiles, self.G.horiz)
        if tile_board[i][(j+1)%n] != -1:
            possible_tiles = self.G.get_compatible_X(possible_tiles, [tile_board[i][(j+1)%n]], self.G.horiz)
        if tile_board[(i+1)%m][j] != -1:
            possible_tiles = self.G.get_compatible_X(possible_tiles, [tile_board[(i+1)%m][j]], self.G.verti)

        # We're moving row by row.
        # TODO: consider other patterns of movements - e.g. filling all the edges of a box (not sure if there's any
        #   difference in speed though...)
        next_j = (j + 1)%n
        next_i = (i + 1)%m if next_j == 0 else i

        for tile_id in tqdm(possible_tiles, disable=not verbose):
            tile_board[i][j] = tile_id

            # Not passing 'verbose' to the next steps on purpose.
            if self.dfs(F, tile_board, next_i, next_j) is not None:
                return tile_board

        # Revert decision for this tile.
        tile_board[i][j] = -1
        return None


class DynamicProg:
    """
    Heuristic approach a bit similar to dynamic programming - unfortunately often doesn't work, since picking a tile
    on one pixel may have "global" effects in a completely different place of the board.
    """
    def __init__(self, tile_graph=None):
        self.G = tile_graph if tile_graph is not None else TileGraph()

    def step_back(self, F, random=False, rseed=12345, verbose=False):
        """
        :param F: final bitmap (boolean matrix)
        :return: previous bitmap (boolean matrix of the same shape as F)
        """

        # Sets of possible previous tiles per central pixel.
        rs = np.random.RandomState(rseed)
        # rs.choice(self.B[F[i][j]], size=100, replace=False) if random else
        S = [[self.G.prev[F[i][j]].copy() for j in range(F.shape[1])] for i in range(F.shape[0])]

        def narrow_down():
            for i in range(m):
                for j in range(n):
                    if len(S[i][j]) > 1:
                        S[i][j] = [S[i][j][0]]
                        if verbose:
                            print(f'Narrowing down {i},{j}')
                        return True
            return False

        m = len(S)
        n = len(S[0])
        while True:
            if verbose:
                print('Loop!')
            changed = False
            for i in range(m):
                for j in range(n):
                    orig_size = len(S[i][j])
                    S[i][j] = self.G.get_compatible_Y(S[i-1][j], S[i][j], self.G.verti)
                    S[i][j] = self.G.get_compatible_Y(S[i][j-1], S[i][j], self.G.horiz)
                    S[i][j] = self.G.get_compatible_X(S[i][j], S[i][(j+1)%n], self.G.horiz)
                    S[i][j] = self.G.get_compatible_X(S[i][j], S[(i+1)%m][j], self.G.verti)

                    #S[i][j] = self.G.get_compatible_X(S[i][j], S[(i+1)%m][(j+1)%n], self.diago_se)
                    #S[i][j] = self.G.get_compatible_X(S[i][j], S[(i+1)%m][j-1], self.diago_sw)
                    #S[i][j] = self.G.get_compatible_Y(S[i-1][j-1], S[i][j], self.diago_se)
                    #S[i][j] = self.G.get_compatible_Y(S[i-1][(j+1)%n], S[i][j], self.diago_sw)
                    if verbose:
                        print(f'{i},{j}: {orig_size} -> {len(S[i][j])}')
                    if len(S[i][j]) == 0:
                        import pdb; pdb.set_trace()
                    if len(S[i][j]) != orig_size:
                        changed = True

            if not changed and not narrow_down():
                break

        # Pick greedily one of the possible configurations of tiles.
        A = np.zeros(F.shape, dtype=np.bool)
        for i in range(m):
            for j in range(n):
                assert len(S[i][j]) == 1
                tile_id = S[i][j][0]

                # Set central bit of the tile to the result bitmap.
                A[i][j] = self.G.tiles[tile_id][1][1]

        return A


class ProbaHeur:
    """
    Heuristic approach similar to DynamicProg. After finding initial plausible tile candidates, it just estimates
    probability of each pixel being '1' separately.
    """
    def __init__(self, tile_graph=None):
        self.G = tile_graph if tile_graph is not None else TileGraph()

    def step_back(self, F, random=False, rseed=12345, verbose=False):
        """
        :param F: final bitmap (boolean matrix)
        :return: previous bitmap (boolean matrix of the same shape as F)
        """

        # Sets of possible previous tiles per central pixel.
        rs = np.random.RandomState(rseed)
        # rs.choice(self.B[F[i][j]], size=100, replace=False) if random else
        S = [[self.G.prev[F[i][j]].copy() for j in range(F.shape[1])] for i in range(F.shape[0])]

        m = len(S)
        n = len(S[0])
        while True:
            if verbose:
                print('Loop!')
            changed = False
            for i in range(m):
                for j in range(n):
                    orig_size = len(S[i][j])
                    S[i][j] = self.G.get_compatible_Y(S[i-1][j], S[i][j], self.G.verti)
                    S[i][j] = self.G.get_compatible_Y(S[i][j-1], S[i][j], self.G.horiz)
                    S[i][j] = self.G.get_compatible_X(S[i][j], S[i][(j+1)%n], self.G.horiz)
                    S[i][j] = self.G.get_compatible_X(S[i][j], S[(i+1)%m][j], self.G.verti)
                    if verbose:
                        print(f'{i},{j}: {orig_size} -> {len(S[i][j])}')
                    assert len(S[i][j]) > 0
                    if len(S[i][j]) != orig_size:
                        changed = True

            if not changed:
                break

        # Pick the most probable pixel on each position.
        A = np.zeros(F.shape, dtype=np.bool)
        for i in range(m):
            for j in range(n):
                tiles = (self.G.tiles[tile_id] for tile_id in S[i][j])
                proba = np.mean([t[1][1] for t in tiles])

                # Set central bit of the tile to the result bitmap.
                A[i][j] = proba > 0.5

        return A


class ProbaHeur2:
    """
    Heuristic approach similar to DynamicProg. After finding initial plausible tile candidates, it just estimates
    probability of each pixel being '1' separately.
    """
    def __init__(self, tile_graph=None):
        self.G = tile_graph if tile_graph is not None else TileGraph()

        self.tile_to_id = {self.G.tiles[i].tobytes(): i for i in range(len(self.G.tiles))}

        self.trans = np.zeros((len(self.G.tiles), len(self.G.tiles)))


    def train(self, delta, start, stop):
        X = start

        m = X.shape[0]
        n = X.shape[1]
        for s in range(delta):
            Y = life_step(X) if s < delta - 1 else stop
            for i in range(m):
                for j in range(n):
                    a_id = self.__get_tile_id(X, i, j)
                    b_id = self.__get_tile_id(Y, i, j)

                    self.trans[a_id][b_id] += 1

    def load_model(self, path):
        with np.load(path) as data:
            self.trans = data['arr_0']

    def save_model(self, path):
        np.savez(path, trans=self.trans)

    def __get_tile_id(self, X, i, j):
        a = np.array(np.roll(np.roll(X, 1-i, axis=0), 1-j, axis=1)[:3,:3], dtype=np.bool)
        a_id = self.tile_to_id[a.tobytes()]
        return a_id

    def predict(self, delta, stop):
        X = stop
        for _ in range(delta):
            X = self.step_back(X, verbose=False)
        return X

    def step_back(self, F, random=False, rseed=12345, verbose=False):
        """
        :param F: final bitmap (boolean matrix)
        :return: previous bitmap (boolean matrix of the same shape as F)
        """

        m = F.shape[0]
        n = F.shape[1]

        # Sets of possible previous tiles per central pixel.
        rs = np.random.RandomState(rseed)
        # rs.choice(self.B[F[i][j]], size=100, replace=False) if random else
        S = [[self.G.prev[F[i][j]].copy() for j in range(n)] for i in range(m)]

        f_tiles = [[self.__get_tile_id(F, i, j) for j in range(n)] for i in range(m)]

        def narrow_down():
            ranking = []
            for i in range(m):
                for j in range(n):
                    if len(S[i][j]) > 1:
                        b_id = f_tiles[i][j]
                        # transition counts to b:
                        trans_counts = np.array([self.trans[s, b_id] for s in S[i][j]])

                        # TODO: hmm it's interesting I haven't seen div by zero here?
                        trans_p = trans_counts / np.sum(trans_counts)
                        max_s_k = np.argmax(trans_p)

                        # TODO: we also use p == 0... should we change that?
                        max_p = S[i][j][max_s_k]
                        ranking.append((max_p,i,j,max_s_k))

            ranking.sort(reverse=True)

            if verbose:
                print(f'{len(ranking)}')

            for r_idx, (p,i,j,k) in enumerate(ranking):
                S[i][j] = [S[i][j][k]]
                if np.abs(p - ranking[0][0]) > 0.0001 and r_idx > len(ranking)//2:
                    break

            return len(ranking) > 0

        while narrow_down():
            if verbose:
                print('Loop!')

            for _ in range(2):
                changed = False

                # narrow down tile based on trans
                for i in range(m):
                    for j in range(n):
                        orig_size = len(S[i][j])
                        S[i][j] = self.G.get_compatible_Y(S[i-1][j], S[i][j], self.G.verti)
                        S[i][j] = self.G.get_compatible_Y(S[i][j-1], S[i][j], self.G.horiz)
                        S[i][j] = self.G.get_compatible_X(S[i][j], S[i][(j+1)%n], self.G.horiz)
                        S[i][j] = self.G.get_compatible_X(S[i][j], S[(i+1)%m][j], self.G.verti)
                        #if verbose:
                        #    print(f'{i},{j}: {orig_size} -> {len(S[i][j])}')
                        if len(S[i][j]) != orig_size:
                            changed = True

                if not changed:
                    break

        # Pick the most probable pixel on each position.
        A = np.zeros(F.shape, dtype=np.bool)
        for i in range(m):
            for j in range(n):
                #assert len(S[i][j]) <= 1
                pixel = self.G.tiles[S[i][j][0]][1][1] if len(S[i][j]) != 0 else False
                # Set central bit of the tile to the result bitmap.
                A[i][j] = pixel

        return A


def train_loop(model_name, learner, early_stop_window=100, rseed=9342184):
    errors = []
    latencies = []
    best_mean_err = 1.0
    best_i = -1
    for i, (delta, start, stop) in enumerate(generate_inf_cases(True, rseed)):
        tic = time.perf_counter()
        A = learner.predict(delta, stop)
        toc = time.perf_counter()

        err = 1 - score(delta, A, stop)
        errors.append(err)

        latency = toc - tic
        latencies.append(latency)

        mean_err = np.mean(errors)
        mean_latency = np.mean(latencies)

        print(f'Error: mean {mean_err}, cur {err}; latency: mean {mean_latency:0.4f}s, cur {latency:0.4f}; delta {delta}, density: {np.mean(stop)}')

        if mean_err < best_mean_err:
            best_mean_err = mean_err
            best_i = i
            file_path = f'{model_name}_{i:05}'
            print(f'    Best model - saving {file_path}...')
            learner.save_model(file_path)
        elif i-best_i > early_stop_window:
            print(f"Haven't found a better model for more than {early_stop_window} iterations - terminating early.")
            print(f"Best iteration: {best_i}, mean error: {best_mean_err}")
            break

        learner.train(delta, start, stop)


if __name__ == '__main__':
    tic = time.perf_counter()
    D = ProbaHeur2()
    toc = time.perf_counter()
    print(f'Initialization: {toc-tic:0.4f}s')

    train_loop('proba_heur2', D)

    """
    for n in [25]:
        print('===========================')
        print(f'Random boards {n}x{n}...')
        for delta, start, stop in generate_inf_cases(True, 9342184):
            tic = time.perf_counter()
            A = stop.copy()
            for i in range(delta):
                A = D.step_back(A, verbose=False)
            toc = time.perf_counter()

            err = 1 - score(delta, A, stop)
            errors.append(err)
            print(f'Acc err: {np.mean(errors)}, now: {err}, time: {toc-tic:0.4f}s; delta: {delta}, dens: {np.mean(stop)}')

            D.train(delta, start, stop)
    """

    """
    Sample output:
    
    Initialization: 4.6644s
    ===========================
    Random boards 3x3...
    Score: 1.0, time: 0.4034s
    Score: 1.0, time: 0.3469s
    Score: 1.0, time: 0.3651s
    Score: 1.0, time: 0.3402s
    Score: 1.0, time: 0.3792s
    Score: 1.0, time: 0.3578s
    Score: 1.0, time: 0.3357s
    Score: 1.0, time: 0.3418s
    Score: 1.0, time: 0.4177s
    Score: 1.0, time: 0.3123s
    ===========================
    Random boards 25x25...
    Score: 0.7776, time: 99.5293s
    Score: 0.7872, time: 93.6009s
    Score: 0.776, time: 131.4938s
    Score: 0.776, time: 95.1731s
    Score: 0.848, time: 99.7578s
    Score: 0.6928, time: 88.6676s
    Score: 0.8624, time: 143.3049s
    Score: 0.8, time: 113.0119s
    Score: 0.7616, time: 118.0421s
    Score: 0.7888, time: 91.3202s
    """