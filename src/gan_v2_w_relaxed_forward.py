"""
GAN assisted by relaxed forward model

    Results after epoch 128:
    [128/25][12797] Loss_D: 0.7455 Loss_G: 0.2303 D(x): 0.2080 D(G(z)): 0.2251 / 0.2372, G MAE: 0.1110
    [128/25][12798] Loss_D: 0.7188 Loss_G: 0.2018 D(x): 0.2083 D(G(z)): 0.2253 / 0.2094, G MAE: 0.0953
    [128/25][12799] Loss_D: 0.7010 Loss_G: 0.2135 D(x): 0.1944 D(G(z)): 0.2091 / 0.2175, G MAE: 0.0962
    [128/25][12800] Loss_D: 0.7089 Loss_G: 0.2040 D(x): 0.2033 D(G(z)): 0.2195 / 0.2137, G MAE: 0.0977
    100%|█████████████████████████████████████████████████████████████████████████████████| 64/64 [00:00<00:00, 2164.54it/s]
    Mean error: one step 0.1054999828338623


"""

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from bitmap import generate_inf_cases
import bitmap
import scoring
from simulator import life_step
from forward_prediction import forward_model




def process_board(board, sigmoid):
    board = board if sigmoid else np.where(board == 0, -1, board)
    return np.array(np.reshape(board, (1, 25, 25)), dtype=np.float32)


class DataGenerator(torch.utils.data.IterableDataset):
    def __init__(self, base_seed, sigmoid):
        super(DataGenerator).__init__()
        self.base_seed = base_seed
        self.sigmoid = sigmoid

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            seed = self.base_seed
        else:  # in a worker process
            # split workload
            worker_id = worker_info.id
            seed = self.base_seed + worker_id
        for delta, prev, stop in generate_inf_cases(True, seed, return_one_but_last=True):
            yield (
                process_board(prev, self.sigmoid),
                process_board(stop, self.sigmoid)
            )


# Validation set
def predict(net, deltas_batch, stops_batch, noise):
    max_delta = np.max(deltas_batch)
    preds = [np.array(stops_batch, dtype=np.float)]
    for _ in range(max_delta):
        tens = torch.Tensor(preds[-1])
        pred_batch = net(tens, noise) > 0.5 #np.array(net(tens) > 0.5, dtype=np.float)
        preds.append(pred_batch)

    # Use deltas as indices into preds.
    # TODO: I'm sure there's some clever way to do the same using numpy indexing/slicing.
    final_pred_batch = []
    for i in range(deltas_batch.size):
        final_pred_batch.append(preds[np.squeeze(deltas_batch[i])][i].detach().numpy())

    return final_pred_batch

def cnnify_batch(batches):
    return (np.expand_dims(batch, 1) for batch in batches)
#


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


# torch.nn.ConvTranspose2d(
#   in_channels: int,
#   out_channels: int,
#   kernel_size: Union[T, Tuple[T, T]],
#   stride: Union[T, Tuple[T, T]] = 1,
#   padding: Union[T, Tuple[T, T]] = 0,
#   output_padding: Union[T, Tuple[T, T]] = 0,
#   groups: int = 1,
#   bias: bool = True,
#   dilation: int = 1,
#   padding_mode: str = 'zeros')
class Generator(nn.Module):
    def __init__(self, ngpu, ngf, nz, use_zgen, sigmoid):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.use_zgen = use_zgen

        ndf = ngf
        self.understand_stop = nn.Sequential(
            # input is (nc) x 25 x 25
            nn.Conv2d(1, ndf, 5, 2, 2, bias=False, padding_mode='circular'),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 13 x 13
            nn.Conv2d(ndf, ndf * 2, 5, 2, 2, bias=False, padding_mode='circular'),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
            # state size. (ndf*2) x 7 x 7
        )

        if use_zgen:
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
            # state size. (ndf*2 + ngf*2) x 7 x 7
            nn.ConvTranspose2d(ndf*2 + zdim,     ngf, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 13 x 13
            nn.ConvTranspose2d(    ngf,      1, 5, 2, 2, bias=False),
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
        z_emb = self.z_gen(z) if self.use_zgen else z.repeat(1, 1, 7, 7)

        # Concatenate channels from stop understanding and z_gen:
        emb = torch.cat([stop_emb, z_emb], dim=1)

        output = self.final_gen(emb)
        return output


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


def train(
        log_dir_prefix,
        device,
        workers,
        batchSize,
        nz,
        ngf,
        niter,
        lr,
        beta1,
        dry_run,
        use_zgen,
        sigmoid=True,
        start_iter=0,
        netGpath='',
        netFpath='',
        outf=None,
        ngpu=1,
        epoch_samples=64*100):

    is_netf = netFpath != ''
    exp_name = f'nz{nz}_ngf{ngf}_zgen{use_zgen}_bs{batchSize}_lr{lr}_beta1{beta1}_netF{is_netf}'
    exp_log_dir = os.path.join(log_dir_prefix, exp_name)

    outf = outf if outf is not None else exp_log_dir
    os.makedirs(outf, exist_ok=True)

    netGpath = netGpath if start_iter == 0 else os.path.join(outf, f'netG_epoch_{start_iter-1}.pth')
    netFpath = netFpath if start_iter == 0 else os.path.join(outf, f'netF_epoch_{start_iter-1}.pth')

    # Prediction threshold
    pred_th = 0.5 if sigmoid else 0.0

    dataset = DataGenerator(823131, sigmoid)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
                                             shuffle=False, num_workers=int(workers))

    val_set = bitmap.generate_test_set(set_size=batchSize, seed=9568382)
    deltas_val, stops_val = cnnify_batch(zip(*val_set))
    ones_val = np.ones_like(deltas_val)

    netG = Generator(ngpu, ngf, nz, use_zgen, sigmoid).to(device)
    netG.apply(weights_init)
    if netGpath != '':
        netG.load_state_dict(torch.load(netGpath))
    print(netG)

    netF = RelaxedForwardNet().to(device)
    netF.apply(weights_init)
    if netFpath != '':
        netF.load_state_dict(torch.load(netFpath))
        print(f'Loaded {netFpath}')
    print(netF)

    criterion = nn.BCELoss()

    fixed_noise = torch.randn(batchSize, nz, 1, 1, device=device)

    # setup optimizer
    optimizerD = optim.Adam(netF.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    scores = []
    for i in range(5):
        noise = torch.randn(batchSize, nz, 1, 1, device=device)
        one_step_pred_batch = (netG(torch.Tensor(stops_val).to(device), noise) > pred_th).cpu()
        model_scores = scoring.score_batch(ones_val, np.array(one_step_pred_batch, dtype=np.bool), stops_val)
        scores.append(model_scores)

    zeros = np.zeros_like(one_step_pred_batch, dtype=np.bool)
    zeros_scores = scoring.score_batch(ones_val, zeros, stops_val)
    scores.append(zeros_scores)

    best_scores = np.max(scores, axis=0)

    print(
        f'Mean error one step: model {1 - np.mean(model_scores)}, zeros {1 - np.mean(zeros_scores)}, ensemble {1 - np.mean(best_scores)}')

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(log_dir=exp_log_dir) if log_dir_prefix else SummaryWriter(comment=exp_name)

    #for epoch in range(opt.niter):
    epoch = start_iter
    samples_in_epoch = 0
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update F (forward) network -- in the original GAN, it's a "D" network (discriminator)
        # Original comment: Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real starting board -- data set provides ground truth
        netF.zero_grad()
        start_real_cpu = data[0].to(device)
        stop_real_cpu = data[1].to(device)
        batch_size = start_real_cpu.size(0)

        output = netF(start_real_cpu)
        errD_real = criterion(output, stop_real_cpu)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake -- use simulator (life_step) to generate ground truth
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(stop_real_cpu, noise)
        fake_np = (fake > pred_th).detach().cpu().numpy()
        fake_next_np = life_step(fake_np)
        fake_next = torch.tensor(fake_next_np, dtype=torch.float32)

        output = netF(fake.detach())
        errD_fake = criterion(output, fake_next.to(device))
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        # just for reporting...
        true_stop_np = (stop_real_cpu > pred_th).detach().cpu().numpy()
        fake_scores = scoring.score_batch(ones_val, fake_np, true_stop_np, show_progress=False)
        fake_mae = 1 - fake_scores.mean()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        output = netF(fake)
        errG = criterion(output, stop_real_cpu)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        writer.add_scalar('Loss/forward', errD.item(), i)
        writer.add_scalar('Loss/gen', errG.item(), i)
        writer.add_scalar('MAE/train', fake_mae.item(), i)

        samples_in_epoch += batch_size
        print('[%d/%d][%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f, G MAE: %.4f'
              % (epoch, niter, i,
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, fake_mae))
        if samples_in_epoch > epoch_samples:
            """
            multi_step_pred_batch = predict(netG, deltas_val, stops_val, fixed_noise)
            multi_step_mean_err = 1 - np.mean(scoring.score_batch(deltas_val, np.array(multi_step_pred_batch, dtype=np.bool), stops_val))
            """

            one_step_pred_batch = (netG(torch.Tensor(stops_val).to(device), fixed_noise) > pred_th).detach().cpu().numpy()
            one_step_mean_err = 1 - np.mean(scoring.score_batch(ones_val, np.array(one_step_pred_batch, dtype=np.bool), stops_val))
            print(f'Mean error: one step {one_step_mean_err}')
            writer.add_scalar('MAE/val', one_step_mean_err, i)

            vutils.save_image(start_real_cpu,
                    '%s/real_samples.png' % outf,
                    normalize=True)
            fake = netG(stop_real_cpu, fixed_noise).detach()
            vutils.save_image(fake,
                    '%s/fake_samples_epoch_%03d.png' % (outf, epoch),
                    normalize=True)

            grid = vutils.make_grid(start_real_cpu)
            writer.add_image('real', grid, i)
            grid = vutils.make_grid(fake)
            writer.add_image('fake', grid, i)

            # do checkpointing
            torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (outf, epoch))
            torch.save(netF.state_dict(), '%s/netF_epoch_%d.pth' % (outf, epoch))
            epoch += 1
            samples_in_epoch = 0

        if epoch - start_iter >= niter:
            break
        if dry_run:
            break

    return one_step_mean_err


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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

    opt = parser.parse_args()
    print(opt)

    exp_name = str(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


    device = torch.device("cuda:0" if opt.cuda else "cpu")
    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)

    train(
        log_dir_prefix=None,
        workers=opt.workers,
        batchSize=opt.batchSize,
        nz=opt.nz,
        ngf=opt.ngf,
        niter=opt.niter,
        lr=opt.lr,
        beta1=opt.beta1,
        dry_run=opt.dry_run,
        netGpath=opt.netG,
        netFpath=opt.netF,
        outf=opt.outf,
        use_zgen=opt.use_zgen,
        sigmoid=opt.sigmoid,
        start_iter=0,
        device=device)