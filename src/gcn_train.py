"""
GAN assisted by relaxed forward model

Best result:

python gcn_train.py --outf gan_sweep_02/fwd_rlx_gen_gan_b2 --fwd_path models/johnson/relaxed_forward.pkl --niter 100 --cuda --gen_arch gan_b

[99/100][9997] Loss_F: 5.2917 Loss_G: 0.3817 fwd acc(real): 1.00 fwd acc(fake): 0.74 / 0.74, fake dens: 0.12, MAE: 0.0645
[99/100][9998] Loss_F: 5.5621 Loss_G: 0.4102 fwd acc(real): 1.00 fwd acc(fake): 0.73 / 0.73, fake dens: 0.13, MAE: 0.0707
[99/100][9999] Loss_F: 5.4020 Loss_G: 0.3661 fwd acc(real): 1.00 fwd acc(fake): 0.75 / 0.75, fake dens: 0.12, MAE: 0.0627
100%|█████████████████████████████████████████████████████████████████████████████| 1024/1024 [00:00<00:00, 2474.15it/s]
Mean error: one step 0.07687658071517944

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

    def forward(self, x, z):
        x = self.activ0(self.conv0(x))
        x = self.activ1(self.conv1(x))
        x = self.activ2(self.conv2(x))
        x = self.activ3(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        return x

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
class GanGenerator(nn.Module):
    def __init__(self, ngpu=1, ngf=64, nz=8, use_zgen=False, sigmoid=True, use_noise=True):
        super(GanGenerator, self).__init__()
        self.ngpu = ngpu
        self.use_zgen = use_zgen
        self.use_noise = use_noise

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

        if self.use_noise:
            z_emb = self.z_gen(z) if self.use_zgen else z.repeat(1, 1, 7, 7)
            # Concatenate channels from stop understanding and z_gen:
            emb = torch.cat([stop_emb, z_emb], dim=1)
        else:
            emb = stop_emb

        output = self.final_gen(emb)
        return output


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


def get_generator_net(gen_arch):
    if gen_arch == 'johnsonC':
        return ReverseNetC()
    elif gen_arch == 'gan':
        return GanGenerator()
    elif gen_arch == 'gan_no_noise':
        return GanGenerator(use_noise=False)
    elif gen_arch == 'gan_zgen':
        return GanGenerator(use_zgen=True)
    elif gen_arch == 'gan_b':
        return GanGeneratorB()
    else:
        raise Exception(f'Unknown generator architecture "{gen_arch}"')


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

def get_forward_net(fwd_arch):
    if fwd_arch == 'relaxed':
        return RelaxedForwardNet()
    else:
        raise Exception(f'Unknown forward architecture "{fwd_arch}"')

def init_model(m, path):
    m.apply(weights_init)
    if path is not None:
        m.load_state_dict(torch.load(path))
        print(f'Loaded {path}')
    print(m)


def train(
        gen_arch,
        fwd_arch,
        device,
        writer,
        batchSize,
        niter,
        lr,
        beta1,
        dry_run,
        outf,
        workers=1,
        start_iter=0,
        gen_path=None,
        fwd_path=None,
        ngpu=1,
        nz=8,
        epoch_samples=64*100,
        learn_forward=True,
        sigmoid=True):

    os.makedirs(outf, exist_ok=True)

    gen_path = gen_path if start_iter == 0 else os.path.join(outf, f'netG_epoch_{start_iter-1}.pth')
    fwd_path = fwd_path if start_iter == 0 else os.path.join(outf, f'netF_epoch_{start_iter-1}.pth')

    # Prediction threshold
    pred_th = 0.5 if sigmoid else 0.0

    dataset = DataGenerator(823131 + start_iter, sigmoid)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
                                             shuffle=False, num_workers=int(workers))

    # uniform
    rand_f = torch.rand

    val_size = 1024
    val_set = bitmap.generate_test_set(set_size=val_size, seed=9568382)
    deltas_val, stops_val = cnnify_batch(zip(*val_set))
    ones_val = np.ones_like(deltas_val)
    noise_val = rand_f(val_size, nz, 1, 1, device=device)

    netG = get_generator_net(gen_arch).to(device)
    init_model(netG, gen_path)

    netF = get_forward_net(fwd_arch).to(device)
    init_model(netF, fwd_path)

    criterion = nn.BCELoss()

    fixed_noise = rand_f(batchSize, nz, 1, 1, device=device)
    fixed_ones = np.ones((batchSize,), dtype=np.int)

    # setup optimizer
    optimizerD = optim.Adam(netF.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    scores = []
    for i in range(5):
        noise = rand_f(val_size, nz, 1, 1, device=device)
        one_step_pred_batch = (netG(torch.Tensor(stops_val).to(device), noise) > pred_th).cpu()
        model_scores = scoring.score_batch(ones_val, np.array(one_step_pred_batch, dtype=np.bool), stops_val)
        scores.append(model_scores)

    zeros = np.zeros_like(one_step_pred_batch, dtype=np.bool)
    zeros_scores = scoring.score_batch(ones_val, zeros, stops_val)
    scores.append(zeros_scores)

    best_scores = np.max(scores, axis=0)

    print(
        f'Mean error one step: model {1 - np.mean(model_scores)}, zeros {1 - np.mean(zeros_scores)}, ensemble {1 - np.mean(best_scores)}')

    #for epoch in range(opt.niter):
    epoch = start_iter
    samples_in_epoch = 0
    samples_before = start_iter * epoch_samples
    for j, data in enumerate(dataloader, 0):
        i = start_iter * epoch_samples // batchSize + j
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
        if learn_forward:
            errD_real.backward()
        D_x = (output.round().eq(stop_real_cpu)).sum().item() / output.numel()

        # train with fake -- use simulator (life_step) to generate ground truth
        # TODO: replace with fixed forward model (should be faster, in batches and on GPU)
        noise = rand_f(batch_size, nz, 1, 1, device=device)
        fake = netG(stop_real_cpu, noise)
        fake_np = (fake > pred_th).detach().cpu().numpy()
        fake_next_np = life_step(fake_np)
        fake_next = torch.tensor(fake_next_np, dtype=torch.float32).to(device)

        output = netF(fake.detach())
        errD_fake = criterion(output, fake_next)
        if learn_forward:
            errD_fake.backward()
        D_G_z1 = (output.round().eq(fake_next)).sum().item() / output.numel()
        errD = errD_real + errD_fake
        if learn_forward:
            optimizerD.step()

        # just for reporting...
        true_stop_np = (stop_real_cpu > pred_th).detach().cpu().numpy()
        fake_scores = scoring.score_batch(fixed_ones, fake_np, true_stop_np, show_progress=False)
        fake_mae = 1 - fake_scores.mean()
        fake_density = fake_np.mean()
        real_density = start_real_cpu.detach().cpu().mean()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        output = netF(fake)
        errG = criterion(output, stop_real_cpu)
        errG.backward()
        D_G_z2 = (output.round().eq(fake_next)).sum().item() / output.numel()
        optimizerG.step()

        samples_in_epoch += batch_size
        s = samples_before + samples_in_epoch
        writer.add_scalar('Loss/forward', errD.item(), i)
        writer.add_scalar('Loss/gen', errG.item(), i)
        writer.add_scalar('MAE/train', fake_mae.item(), i)
        writer.add_scalar('Fwd accuracy/real', D_x, i)
        writer.add_scalar('Fwd accuracy/fake_unseen', D_G_z1, i)
        writer.add_scalar('Fwd accuracy/fake_seen', D_G_z2, i)
        writer.add_scalar('Density/real_start', real_density, i)
        writer.add_scalar('Density/fake_start', fake_density, i)
        print('[%d/%d][%d] Loss_F: %.4f Loss_G: %.4f fwd acc(real): %.2f fwd acc(fake): %.2f / %.2f, fake dens: %.2f, MAE: %.4f'
              % (epoch, start_iter+niter, i,
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, fake_density, fake_mae))
        if samples_in_epoch >= epoch_samples:
            """
            multi_step_pred_batch = predict(netG, deltas_val, stops_val, fixed_noise)
            multi_step_mean_err = 1 - np.mean(scoring.score_batch(deltas_val, np.array(multi_step_pred_batch, dtype=np.bool), stops_val))
            """

            one_step_pred_batch = (netG(torch.Tensor(stops_val).to(device), noise_val) > pred_th).detach().cpu().numpy()
            one_step_mean_err = 1 - np.mean(scoring.score_batch(ones_val, np.array(one_step_pred_batch, dtype=np.bool), stops_val))
            print(f'Mean error: one step {one_step_mean_err}')
            writer.add_scalar('MAE/val', one_step_mean_err, epoch)

            vutils.save_image(start_real_cpu,
                    '%s/real_samples.png' % outf,
                    normalize=True)
            fake = netG(stop_real_cpu, fixed_noise).detach()
            vutils.save_image(fake,
                    '%s/fake_samples_epoch_%03d.png' % (outf, epoch),
                    normalize=True)

            grid = vutils.make_grid(start_real_cpu)
            writer.add_image('real', grid, epoch)
            grid = vutils.make_grid(fake)
            writer.add_image('fake', grid, epoch)

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
    parser.add_argument('--outf', required=True, help='folder to output images, model checkpoints and tensorboard logs')

    parser.add_argument('--fwd_arch', default='relaxed')
    parser.add_argument('--gen_arch', default='gan')

    parser.add_argument('--fwd_path', default=None, help="path to netF (to continue training)")
    parser.add_argument('--gen_path', default=None, help="path to netG (to continue training)")

    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--start_iter', type=int, default=0)

    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    parser.add_argument('--improve_fwd', action='store_true')

    opt = parser.parse_args()
    print(opt)

    exp_name = str(opt)

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = torch.device("cuda:0" if opt.cuda else "cpu")

    writer = SummaryWriter(log_dir=opt.outf)

    train(
        gen_arch=opt.gen_arch,
        fwd_arch=opt.fwd_arch,
        gen_path=opt.gen_path,
        fwd_path=opt.fwd_path,
        writer=writer,
        outf=opt.outf,
        start_iter=opt.start_iter,
        workers=opt.workers,
        batchSize=opt.batchSize,
        niter=opt.niter,
        lr=opt.lr,
        beta1=opt.beta1,
        dry_run=opt.dry_run,
        device=device,
        learn_forward=opt.improve_fwd)