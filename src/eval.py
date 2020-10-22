import numpy as np
import argparse
import bitmap
import baselines
import tile_graph
from scoring import score
from tabulate import tabulate
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiment evaluations for the JJS229 project.')

    parser.add_argument('--all', action='store_true', help='Run all experiment evaluations.')

    parser.add_argument('--baselines', action='store_true', help='Run baseline evaluations.')
    parser.add_argument('--proba_heur2_path', action='append', help='Run the ProbaHeur2 model from the provided paths.')

    parser.add_argument('--test_seed', type=int, default=9568382, help='Random seed for test set generation.')
    parser.add_argument('--test_size', type=int, default=10000, help='Test set size.')
    args = parser.parse_args()

    print(f'Arguments: {args}')

    def eval(predict):
        multi_step_errors = []
        one_step_errors = []
        for delta, stop in tqdm(bitmap.generate_test_set(set_size=args.test_size, seed=args.test_seed)):
            start = predict(delta, stop)
            multi_step_errors.append(1 - score(delta, start, stop))

            one_step_start = start if delta == 1 else predict(1, stop)
            one_step_errors.append(1 - score(1, one_step_start, stop))
        return np.mean(multi_step_errors), np.var(multi_step_errors), np.mean(one_step_errors), np.var(one_step_errors)

    model_names = []
    models = []

    if args.all or args.baselines:
        model_names.extend(['const_zeros', 'mirror', 'random_tries'])
        models.extend([baselines.const_zeros, baselines.mirror, baselines.random_tries])

    if args.proba_heur2_path is not None:
        for path in args.proba_heur2_path:
            model_names.append(path)
            m = tile_graph.ProbaHeur2()
            m.load_model(path)
            models.append(m.predict)

    data = []
    for model_name, model in zip(model_names, models):
        multi_step_mean, multi_step_var, one_step_mean, one_step_var = eval(model)
        data.append((model_name, multi_step_mean, multi_step_var, one_step_mean, one_step_var))

    print(tabulate(data, headers=['model', 'multi-step mean', 'multi-step var', 'one step mean', 'one step var'], tablefmt='orgtbl'))
