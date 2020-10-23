import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
from collections import defaultdict
from itertools import product
from tqdm import tqdm

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

def life_step(X):
    """Game of life step using generator expressions"""
    nbrs_count = sum(np.roll(np.roll(X, i, 0), j, 1)
                     for i in (-1, 0, 1) for j in (-1, 0, 1)
                     if (i != 0 or j != 0))
    return (nbrs_count == 3) | (X & (nbrs_count == 2))

def generate():
    evolved_board = np.zeros((25, 25), dtype=int)
    seed_board = None
    i = 0
    
    while np.sum(evolved_board) == 0 and i < 5:
        i += 1
        # Create seeding board
        seed_board = np.random.randint(0, 2, (25, 25))
        # Warm up by evolving 5 steps
        for _ in range(5):
            seed_board = life_step(seed_board)
        evolved_board = life_step(seed_board)
        
    return seed_board, evolved_board

def get_wrapped(matrix, i, j):
    m, n = matrix.shape
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

    predictions = np.zeros((n, 25, 25), dtype=int)
    deltas = df.delta.to_list()

    print("Making predictions on the test data")
    for i in tqdm(range(n)):
        best_guess = np.zeros((25, 25), dtype=int)
        mae = float('inf')
        for _ in range(num_tries):
            end_board = stopping_boards[i]
            for _ in range(deltas[i]):
                guess = np.zeros((25, 25), dtype=int)
                for j in range(25):
                    for k in range(25):
                        prob_on = model[flatten_cell(get_wrapped(end_board, j, k))]
                        if np.random.rand() <= prob_on:
                            guess[j, k] = 1
                end_board = np.copy(guess)
            # Check whether the guess if the best so far
            for _ in range(deltas[i]):
                end_board = life_step(end_board)
            mae_ = np.sum(stopping_boards[i] != end_board) / 625
            if mae_ < mae:
                best_guess = guess
                mae = mae_
        predictions[i] = best_guess
        
        if i % 100 == 0 and i > 0:
            print("iteration: {}, mae: {}".format(i, mae))
            
    df_ = df[['id']].copy()
    predictions = predictions.reshape(n, 625)
    for i in range(625):
        df_['start_{}'.format(i)] = predictions[:,i]
    return df_

if __name__ == '__main__':
    N = 200000
    new_data_path = '../../data/new_data.csv'
    model_path = '../../data/model.pkl'
    test_data_path = '../../data/test.csv'
    submission_path = '../../data/submissions.csv'
    with open(new_data_path, 'w') as outfile:
        # Write column header
        columns = ['start_{}'.format(i) for i in range(625)] + ['stop_{}'.format(i) for i in range(625)]
        outfile.write(','.join(columns))
        outfile.write('\n')
        for i in range(N):
            start_board, stop_board = generate()
            row = ','.join(map(str, start_board.flatten())) + ','
            row += ','.join(map(str, stop_board.flatten())) + '\n'
            outfile.write(row)
    
    model = create_prob_model(new_data_path, model_path)
    #model = pickle.load(open(model_path, 'rb'))
    predictions = predict(test_data_path, model) 
    predictions.to_csv(submission_path, index=False)
