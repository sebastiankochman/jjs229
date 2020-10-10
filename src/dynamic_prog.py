# Algorithm based on dynamic programming

import numpy as np
from simulator import life_step
from bitmap import generate_bitmaps


class DynamicProg:
    def __init__(self):
        # T - all possible titles 3x3
        # B - backward possibilities, i.e., dict: central bit -> list of possible prev tiles 3x3
        # Matrix R[i,j] - true if tile j can be put on the right side of tile i.
        # Matrix D[i,j] - true if tile j can be put under tile i ('Down')
        self.T, self.B, self.R, self.D = self.preprocess()

    def step_back(self, F, verbose=False):
        """
        :param F: final bitmap (boolean matrix)
        :return: previous bitmap (boolean matrix of the same shape as F)
        """

        # Sets of possible previous tiles per central pixel.
        S = [[self.B[F[i][j]].copy() for j in range(F.shape[1])] for i in range(F.shape[0])]

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
                    S[i][j] = self.get_compatible_Y(S[i-1][j], S[i][j], self.D)
                    S[i][j] = self.get_compatible_Y(S[i][j-1], S[i][j], self.R)
                    S[i][j] = self.get_compatible_X(S[i][j], S[i][(j+1)%n], self.R)
                    S[i][j] = self.get_compatible_X(S[i][j], S[(i+1)%m][j], self.D)
                    if verbose:
                        print(f'{i},{j}: {orig_size} -> {len(S[i][j])}')
                    if len(S[i][j]) != orig_size:
                        changed = True

            if not changed and not narrow_down():
                break

        # Pick greedily one of the possible configurations of tiles.
        A = np.zeros(F.shape, dtype=np.bool)
        for i in range(len(S)):
            for j in range(len(S[i])):
                assert len(S[i][j]) == 1
                tile_id = S[i][j][0]

                # Set central bit of the tile to the result bitmap.
                A[i][j] = self.T[tile_id][1][1]

        return A

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

        # Matrix R[i,j] - true if tile j can be put on the right side of tile i.
        R = np.zeros((len(T), len(T)), dtype=np.bool)
        for i, left in enumerate(T):
            for j, right in enumerate(T):
                # Check if left tile is compatible with the right tile.
                # I.e., left tile's right side is equal to right tile's left side.
                R[i, j] = (left[:, (1, 2)] == right[:, (0, 1)]).all()

        # Matrix D[i,j] - true if tile j can be put under tile i ('Down')
        D = np.zeros((len(T), len(T)), dtype=np.bool)
        for i, upper in enumerate(T):
            for j, lower in enumerate(T):
                # Check if upper tile is compatible with the lower tile.
                # I.e., upper tile's lower side is equal to lower tile's upper side.
                D[i, j] = (upper[(1, 2), :] == lower[(0, 1), :]).all()

        return T, B, R, D

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
