# Algorithm based on dynamic programming

import numpy as np
import time
from simulator import life_step
from bitmap import generate_bitmaps
from scoring import score
from tqdm import tqdm

class TileGraph:
    def __init__(self):
        self.tiles, self.prev, self.horiz, self.verti = self.preprocess()

    @staticmethod
    def preprocess():
        # Tiles - all possible titles 3x3
        T = list(generate_bitmaps(3, 3))
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



if __name__ == '__main__':
    tic = time.perf_counter()
    D = DFS()
    toc = time.perf_counter()
    print(f'Initialization: {toc-tic:0.4f}s')

    # 8x8 is already too much for DFS.
    for n in range(3, 8):
        print(f'{n}x{n}...')
        X = np.random.choice([0, 1], size=(n, n))
        for i in range(6):
            X = life_step(X)
        tic = time.perf_counter()
        A = D.step_back(X, verbose=True)
        toc = time.perf_counter()
        print(f'Full step back: {toc-tic:0.4f}s')
