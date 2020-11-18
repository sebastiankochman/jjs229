"""
Results:

python gcn_eval.py --gen_path models/gans/20201116_v02/netG_epoch_351.pth --cuda --baselines


| model            |   multi-step mean |   multi-step var |   one step mean |   one step var |
|------------------+-------------------+------------------+-----------------+----------------|
| const_zeros      |          0.147179 |       0.00792056 |       0.147179  |     0.00792056 |
| mirror           |          0.172068 |       0.0127246  |       0.129641  |     0.00755696 |
| likely_starts    |          0.141522 |       0.00828948 |       0.128372  |     0.00759826 |
| gcn              |          0.149215 |       0.0119487  |       0.0634629 |     0.00229834 |
| gcn_multi        |          0.141785 |       0.0107674  |       0.0615779 |     0.0021672  |
| gcn+zeros        |          0.125545 |       0.00774776 |       0.0634882 |     0.00230135 |
| gcn+likely       |          0.124965 |       0.00788405 |       0.0633694 |     0.00231712 |
| gcn_multi+likely |          0.123533 |       0.00780647 |       0.0614331 |     0.00218183 |
"""



from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import argparse
import bitmap
import baselines
from scoring import score_batch, score
from tabulate import tabulate
from tqdm import tqdm
import itertools
import random

# Johnson's:
class RelaxedForwardNet(nn.Module):
    def __init__(self):
        super(RelaxedForwardNet, self).__init__()
        # in channels, out channels, kernel size
        self.conv0 = nn.Conv2d(1, 8, (1, 1))
        self.activ0 = nn.ReLU()
        self.conv1 = nn.Conv2d(8, 16, (3, 3), padding=(1, 1), padding_mode='circular')
        self.activ1 = nn.PReLU()
        self.conv2 = nn.Conv2d(16, 8, (3, 3), padding=(1, 1), padding_mode='circular')
        self.activ2 = nn.PReLU()
        self.conv3 = nn.Conv2d(8, 4, (3, 3), padding=(1, 1), padding_mode='circular')
        self.activ3 = nn.PReLU()
        self.conv4 = nn.Conv2d(4, 1, (3, 3), padding=(1, 1), padding_mode='circular')

    def forward(self, x):
        x = self.activ0(self.conv0(x))
        x = self.activ1(self.conv1(x))
        x = self.activ2(self.conv2(x))
        x = self.activ3(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        return x

class ReverseNetC(nn.Module):
    def __init__(self):
        super(ReverseNetC, self).__init__()
        # in channels, out channels, kernel size
        self.conv0 = nn.Conv2d(1, 8, (1, 1))
        self.activ0 = nn.ReLU()
        self.conv1 = nn.Conv2d(8, 16, (3, 3), padding=(1, 1), padding_mode='circular')
        self.activ1 = nn.PReLU()
        self.conv2 = nn.Conv2d(16, 8, (3, 3), padding=(1, 1), padding_mode='circular')
        self.activ2 = nn.PReLU()
        self.conv3 = nn.Conv2d(8, 4, (3, 3), padding=(1, 1), padding_mode='circular')
        self.activ3 = nn.PReLU()
        self.conv4 = nn.Conv2d(4, 1, (3, 3), padding=(1, 1), padding_mode='circular')

    def forward(self, x, z=None):
        x = self.activ0(self.conv0(x))
        x = self.activ1(self.conv1(x))
        x = self.activ2(self.conv2(x))
        x = self.activ3(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        return x

class ReverseForwardNet(nn.Module):
    def __init__(self, ForwardNet, forward_wt_path, ReverseNet):
        super(ReverseForwardNet, self).__init__()
        self.reverse_net = ReverseNet()
        # freeze the weights of the forward net
        self.forward_net = ForwardNet()
        self.forward_net.load_state_dict(torch.load(forward_wt_path))
        for param in self.forward_net.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.reverse_net(x)
        x = self.forward_net(x)
        return x

# /Johnson's


class GanGeneratorB(nn.Module):
    def __init__(self, ngpu=1, ngf=64, nz=8, use_zgen=False, sigmoid=True, use_noise=True):
        super(GanGeneratorB, self).__init__()
        self.ngpu = ngpu
        self.use_zgen = use_zgen
        self.use_noise = use_noise

        ndf = ngf
        self.understand_stop = nn.Sequential(
            # input is (nc) x 25 x 25
            nn.Conv2d(1, ndf, 5, 1, 2, bias=False, padding_mode='circular'),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 25 x 25
            nn.Conv2d(ndf, ndf * 2, 5, 1, 2, bias=False, padding_mode='circular'),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
            # state size. (ndf*2) x 25 x 25
        )

        if not use_noise:
            zdim = 0
        elif use_zgen:
            self.z_gen = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(     nz, ngf * 4, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 4 x 4
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 5, 2, 2, bias=False), # padding_mode='circular' not available in ConvTranspose2d
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True)
                # state size. (ngf*2) x 7 x 7
            )
            zdim = ngf * 2
        else:
            zdim = nz

        self.final_gen = nn.Sequential(
            # state size. (ndf*2 + ngf*2) x 25 x 25
            nn.ConvTranspose2d(ndf*2 + zdim,     ngf, 5, 1, 2, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 25 x 25
            nn.ConvTranspose2d(    ngf,      1, 5, 1, 2, bias=False),
            nn.Sigmoid() if sigmoid else nn.Tanh()
            # state size. (nc) x 25 x 25
        )


    def forward(self, stop, z):
        # TODO: fix CUDA:
        #if input.is_cuda and self.ngpu > 1:
        #    output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        #else:

        # Concatenate channels from stop understanding and z_gen:
        stop_emb = self.understand_stop(stop)

        if self.use_noise:
            z_emb = self.z_gen(z) if self.use_zgen else z.repeat(1, 1, 25, 25)
            # Concatenate channels from stop understanding and z_gen:
            emb = torch.cat([stop_emb, z_emb], dim=1)
        else:
            emb = stop_emb

        output = self.final_gen(emb)
        return output


def init_model(m, path):
    if path is not None:
        m.load_state_dict(torch.load(path))
        print(f'Loaded {path}')
    print(m)


def predict(net, deltas_batch, stops_batch):
    max_delta = torch.max(deltas_batch)
    preds = [stops_batch]
    for _ in range(max_delta):
        nz = 8
        noise = torch.rand(deltas_batch.shape[0], nz, 1, 1, device=device)
        pred_batch = torch.tensor(net(preds[-1], noise) > 0.5, dtype=torch.float, device=device)
        preds.append(pred_batch)

    # Use deltas as indices into preds.
    # TODO: I'm sure there's some clever way to do the same using numpy indexing/slicing.
    final_pred_batch = []
    for i in range(deltas_batch.shape[0]):
        final_pred_batch.append(preds[deltas_batch[i].item()][i].detach().cpu().numpy() > 0.5)

    return np.array(final_pred_batch, dtype=np.bool)


def ensemble(predicts, deltas_batch, stops_batch):
    predictions = np.array([p(deltas_batch, stops_batch) for p in predicts])
    scores = np.array([[score(deltas_batch[j], predictions[i][j], stops_batch[j]) for j in range(len(predictions[i]))] for i in range(len(predicts))])
    max_idxs = np.argmax(scores, axis=0)
    best = []
    for i,best_idx in enumerate(max_idxs):
        best.append(predictions[best_idx][i])
    return np.array(best)


def cnnify_batch(batches):
    return (np.expand_dims(batch, 1) for batch in batches)


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiment evaluations for the JJS229 project.')

    parser.add_argument('--all', action='store_true', help='Run all experiment evaluations.')

    parser.add_argument('--baselines', action='store_true', help='Run baseline evaluations.')
    parser.add_argument('--gen_path', required=True, help="path to netG")
    parser.add_argument('--johnson', action='store_true', help='evaluates Johnsons reverse-forward model')

    parser.add_argument('--test_seed', type=int, default=9568382, help='Random seed for test set generation.')
    parser.add_argument('--test_size', type=int, default=10000, help='Test set size.')

    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    args = parser.parse_args()

    print(f'Arguments: {args}')

    # To make results reproducable:
    random.seed(8912891)
    torch.manual_seed(317218)

    def eval(predict):
        multi_step_errors = []
        one_step_errors = []
        for batch in tqdm(grouper(bitmap.generate_test_set(set_size=args.test_size, seed=args.test_seed), 100)):
            deltas, stops = zip(*batch)

            delta_batch = np.array(deltas)
            stop_batch = np.array(stops)
            start_batch = predict(deltas, stops)

            for delta, start, stop in zip(delta_batch, start_batch, stop_batch):
                multi_step_errors.append(1 - score(delta, start, stop))

            one_deltas = np.ones_like(delta_batch)
            one_step_start = np.where(deltas == 1, start_batch, predict(one_deltas, stops))
            for delta, start, stop in zip(one_deltas, one_step_start, stop_batch):
                one_step_errors.append(1 - score(delta, start, stop))
        return np.mean(multi_step_errors), np.var(multi_step_errors), np.mean(one_step_errors), np.var(one_step_errors)

    model_names = []
    models = []

    def batchify(f):
        def run_batch(d, s):
            r = []
            for i in range(len(d)):
                r.append(f(d[i], s[i]))
            return np.array(r)
        return run_batch

    if args.all or args.baselines:
        model_names.extend(['const_zeros', 'mirror', 'likely_starts'])
        models.extend([batchify(baselines.const_zeros), batchify(baselines.mirror), batchify(baselines.likely_starts)])

    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.johnson:
        netG = ReverseForwardNet(RelaxedForwardNet, args.gen_path, ReverseNetC).reverse_net.to(device)
    else:
        netG = GanGeneratorB().to(device)
        init_model(netG, args.gen_path)

    def gcn_predict(deltas, stops):
        deltas_batch = torch.tensor(deltas).to(device)
        stop_batch = torch.tensor(np.array(np.expand_dims(stops, 1), dtype=np.float32)).to(device)
        return predict(netG, deltas_batch, stop_batch).squeeze()

    def gcn_multi(deltas, stops):
        return ensemble([gcn_predict, gcn_predict, gcn_predict, gcn_predict, gcn_predict], deltas, stops)

    def gcn_plus_zeros(deltas, stops):
        return ensemble([gcn_predict, batchify(baselines.const_zeros)], deltas, stops)

    def gcn_plus_likely(deltas, stops):
        return ensemble([gcn_predict, batchify(baselines.likely_starts)], deltas, stops)

    def gcn_multi_plus_likely(deltas, stops):
        return ensemble([gcn_predict, gcn_predict, gcn_predict, gcn_predict, gcn_predict, batchify(baselines.likely_starts)], deltas, stops)

    model_names.extend(['R', 'R_multi', 'R+zeros', 'R+likely', 'R_multi+likely'])
    models.extend([gcn_predict, gcn_multi, gcn_plus_zeros, gcn_plus_likely, gcn_multi_plus_likely])

    data = []
    for model_name, model in zip(model_names, models):
        multi_step_mean, multi_step_var, one_step_mean, one_step_var = eval(model)
        data.append((model_name, f'{one_step_mean*100:.2f}%', f'{one_step_var*100:.2f}%', f'{multi_step_mean*100:.2f}%', f'{multi_step_var*100:.2f}%'))

    print(tabulate(data, headers=['model', 'one step mean', 'one step var', 'multi-step mean', 'multi-step var'], tablefmt='orgtbl'))


    print(tabulate(data, headers=['model', 'one step mean', 'one step var', 'multi-step mean', 'multi-step var'], tablefmt='latex'))

