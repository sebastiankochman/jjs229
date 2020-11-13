import numpy as np

def generate():
    # Create seeding board
    seed_board = np.random.randint(0, 2, (25, 25)) 

    # Warm up by evolving 5 steps
    for _ in range(5):
        seed_board = life_step(seed_board)
   
    evolved_board = seed_board.copy()
    num_steps = np.random.randint(1, 6)
    for _ in range(num_steps):
        evolved_board = life_step(evolved_board)
  
    final_board = life_step(evolved_board)

    if np.sum(final_board) > 0:
        return num_steps, evolved_board, final_board

def life_step(X):
    """Game of life step using generator expressions"""
    nbrs_count = sum(np.roll(np.roll(X, i, 0), j, 1)
                     for i in (-1, 0, 1) for j in (-1, 0, 1)
                     if (i != 0 or j != 0))
    return (nbrs_count == 3) | (X & (nbrs_count == 2))

if __name__ == '__main__':
    N = 100000
    with open('../../data/extra.csv', 'w') as outfile:
        for i in range(N): 
            num_steps, start_board, stop_board = generate()
            #row = f'{i}, {num_steps}, '
            row = ', '.join(map(str, start_board.flatten())) + ', '
            row += ', '.join(map(str, stop_board.flatten())) + '\n'
            outfile.write(row)
