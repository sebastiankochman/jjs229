import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from itertools import product
from tqdm import tqdm


def life_step(X):
    """Game of life step using generator expressions"""
    nbrs_count = sum(np.roll(np.roll(X, i, 0), j, 1)
                     for i in (-1, 0, 1) for j in (-1, 0, 1)
                     if (i != 0 or j != 0))
    return (nbrs_count == 3) | (X & (nbrs_count == 2))


def get_wrapped(matrix, i, j):
  m, n = matrix.shape
  # rows = [(i-2) % m, (i-1) % m, i, (i+1) % m, (i+2) % m]
  # cols = [(j-2) % n, (j-1) % n, j, (j+1) % n, (j+2) % n]
  rows = [(i-1) % m, i, (i+1) % m]
  cols = [(j-1) % n, j, (j+1) % n]
  return matrix[rows][:, cols]


def flatten_cell(cell):
    return ''.join(map(str, cell.flatten()))


def create_prob_model(input_path, output_path):
    df = pd.read_csv(input_path, ',')
    n = df.shape[0]
    starting_cols = [col for col in df.columns if col.startswith('start')]
    starting_boards = df[starting_cols].to_numpy().reshape(n, 25, 25)
    stopping_cols = [col for col in df.columns if col.startswith('stop')]
    stopping_boards = df[stopping_cols].to_numpy().reshape(n, 25, 25)

    dict_on = {''.join(map(str, l)): 0 for l in product(range(2), repeat=9)}
    dict_off = {''.join(map(str, l)): 0 for l in product(range(2), repeat=9)}
    
    print("Creating the probabilistic model")
    for i in tqdm(range(n)):
        for j in range(25):
            for k in range(25):
                evolved_block = flatten_cell(get_wrapped(stopping_boards[i], j, k))
                starting_cell = starting_boards[i, j, k]
                if starting_cell == 1:
                    dict_on[evolved_block] += 1
                else:
                    dict_off[evolved_block] += 1
    
    for k in dict_on:
        s = dict_on[k] + dict_off[k]
        dict_on[k], dict_off[k] = dict_on[k] / s, dict_off[k] / s

    with open(output_path, 'wb') as outfile:
        pickle.dump(dict_on, outfile)

    return dict_on


def predict(input_data, model, num_tries=5):
    df = pd.read_csv(input_data, ',')
    n = df.shape[0]
    cols = [col for col in df.columns if col.startswith('stop')]
    stopping_boards = df[cols].to_numpy().reshape(n, 25, 25)

    predictions = np.zeros((n, 25, 25))
    deltas = df.delta.to_list()

    print("Making predictions on the test data")
    for i in tqdm(range(n)):
        best_guess = np.zeros((25, 25), dtype=int)
        mae = float('inf')
        for _ in range(num_tries):
            prev_board = np.zeros((25, 25), dtype=int)
            end_board = stopping_boards[i]
            for _ in range(deltas[i]):
                for j in range(25):
                    for k in range(25):
                        prob_on = model[flatten_cell(get_wrapped(end_board, j, k))]
                        if np.random.rand() <= prob_on:
                            prev_board[j, k] = 1
                end_board = np.copy(prev_board)
            guess = np.copy(prev_board)
            # Check whether the guess if the best so far
            for _ in range(deltas[i]):
                end_board = life_step(end_board)
            mae_ = np.sum(stopping_boards[i] != end_board)
            if mae_ < mae:
                best_guess = guess
        predictions[i] = best_guess
   
    df_ = df[['id']]
    predictions = predictions.reshape(n, 625)
    for i in range(625):
        df_['start_{}'.format(i)] = predictions[:,i]
    return df_

if __name__ == '__main__':
    # model = create_prob_model('../../data/extra.csv', '../../data/model.pkl')
    model = pickle.load(open('../../data/model.pkl', 'rb'))
    predictions = predict('../../data/test.csv', model) 
    predictions.to_csv('../../data/submission.csv')
