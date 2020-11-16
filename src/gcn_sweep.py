import torch
from gcn_train import train

"""
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netF', default='', help="path to netF (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--sigmoid', action='store_true', help='use sigmoid activation in generator')
parser.add_argument('--use_zgen', action='store_true', help='use zgen')
"""

"""
p = {
    'batchSize': [64],
    'nz': [10, 50],
    'ngf': [32],
    'lr': [0.0002],
    'beta1': [0.5],
    'use_zgen': [True, False],
    'netFpath': ['models/johnson/relaxed_forward.pkl']  # skipped: 'models/johnson/forward_with_relaxation.pkl'
}

"""
p = {
    'batchSize': [64],
    'nz': [10, 50, 100],
    'ngf': [32, 64, 128],
    'lr': [0.0002],
    'beta1': [0.5],
    'use_zgen': [True, False],
    'netFpath': ['models/johnson/relaxed_forward.pkl', '']  # skipped: 'models/johnson/forward_with_relaxation.pkl'
}


combinations = []

def generate_combinations(config, keys_left):
    if len(keys_left) == 0:
        combinations.append(dict(config))
    else:
        k = keys_left.pop()
        for val in p[k]:
            config[k] = val
            generate_combinations(config, keys_left)
        keys_left.add(k)

keys = set(p.keys())
generate_combinations({}, keys)

print(f'Generated {len(combinations)} param combinations')

survivors = set(range(len(combinations)))

epoch = 0
start_iter = 0
niter_per_epoch = 2
cuda = True

device = torch.device("cuda:0" if cuda else "cpu")

while len(survivors) > 1:
    maes = {}
    for i in survivors:
        params = combinations[i]
        print('=====================================')
        print(f'Training with params #{i}: {params}...')

        # TODO: temporarily broken (changed the train function recently)
        mae = train(
            log_dir_prefix='gan_sweep_01',
            workers=2,
            niter=niter_per_epoch,
            dry_run=False,
            sigmoid=True,
            start_iter=start_iter,
            device=device,
            **params)
        maes[i] = mae

    ordered = sorted(survivors, key=lambda x: maes[x])
    print(f'\n\n==================================================\n\nMAE:\n{maes}\n\nExperiments ordered by MAE:\n{ordered}')
    survivors = set(ordered[:((len(ordered)+1)//2)])
    print(f'\n\n------------------------------------------------\n\nNew survivors:\n\n{survivors}\n\n')

    start_iter += niter_per_epoch
    niter_per_epoch *= 2
    epoch += 1