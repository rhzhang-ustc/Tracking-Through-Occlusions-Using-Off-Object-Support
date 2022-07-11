import time
import numpy as np
import matplotlib
import argparse


matplotlib.use('Agg')  # suppress plot showing
import utils.py
import utils.misc
import utils.improc
import utils.grouping
import utils.samp
import utils.basic
import random

from utils.basic import print_, print_stats

# import datasets
import flyingthingsdataset

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from tensorboardX import SummaryWriter

import torch.nn.functional as F

from perceiver_io import PerceiverIO
from  position_encoding import generate_fourier_features

device = 'cuda'
patch_size = 8
random.seed(125)
np.random.seed(125)

## choose hyps
B = 1
S = 8
N = 64 +1 # we need to load at least 4 i think
lr = 1e-4
grad_acc = 1

crop_size = (368, 496)
crop_size_3d = (368, 496, S)

max_iters = 10000
log_freq = 2
save_freq = 1000
shuffle = True
do_val = False
cache = True
cache_len = 100
cache_freq = 99999999
use_augs = False

val_freq = 10

queries_dim = 32
dim = 3
feature_dim = 27
logis_dim = 2
num_band = 64

log_dir = 'logs'

def extract_frame_patches(frames, trajs, step=1):
    # assert B = 1
    # steps: dilate conv style
    # frames: (S, C, H, W)
    # trajs: (S, N, 2)
    # output: (S, N, feature_length) where feature are extracted according to (x, y)

    S, C, H, W = frames.shape
    _, N, _ = trajs.shape

    # generate grid matrix for flow, B*S, N*9, 2
    x = torch.tensor(trajs[:, :, 0], dtype=torch.int).unsqueeze(-1)  # B, N, 1
    y = torch.tensor(trajs[:, :, 1], dtype=torch.int).unsqueeze(-1)

    x_range = torch.concat([x-step, x, x+step], axis=-1).unsqueeze(-1)  # B, N, 3, 1
    y_range = torch.concat([y-step, y, y+step], axis=-1).unsqueeze(-2)  # B, N, 1, 3

    grid_x = x_range.repeat(1, 1, 1, 3).flatten(start_dim=-2)  # meshgrid, B, N, 9
    grid_y = y_range.repeat(1, 1, 3, 1).flatten(start_dim=-2)

    # bilinear sample, B*S, C, N*3*3
    output = torch.zeros((B*S, C, N, 9))
    for i in range(3):
        for j in range(3):
            output[:, :, :, 3*i+j] = utils.samp.bilinear_sample2d(frames, grid_x[:, :, i], grid_y[:, :, j])

    # reshape, S, N, C*(2*step+1)*(2*step+1)
    output = output.transpose(1, 2).reshape(S, N, -1)

    return output


def run_model(model, queries, sample, criterion):

    total_loss = torch.tensor(0.0, requires_grad=True).to(device)

    rgbs = torch.from_numpy(sample['rgbs']).cuda().float()  # B, S, C, H, W
    trajs = sample['trajs'].cuda().float()  # B, S, N, 2

    B, S, C, H, W = rgbs.shape
    _, _, N, _ = trajs.shape

    rgbs = rgbs.reshape(-1, C, H, W)  # S, C, H, W
    trajs = trajs.reshape(-1, N, 2)     # S, N, 2

    t_pos = torch.arange(0, S).cuda().float()
    t_pos = t_pos.expand([N, -1]).unsqueeze(-1)
    t_pos = t_pos.transpose(0, 1)  # S, N, 1
    pos = torch.concat([trajs, t_pos], axis=-1)  # S, N, 3

    # generate target: one trajectory that needs to be estimate
    # then delete the point from trajs

    target_traj = trajs[:, 0, :].unsqueeze(-2)  # S, 1, 2

    target_pos = pos[:, 0, :].unsqueeze(-2)
    pos = pos[:, 1:, :]  # S, N-1, 3
    relative_pos = pos - target_pos

    trajs = trajs[:, 1:, :]
    traj_encoding = generate_fourier_features(relative_pos.reshape(-1, dim).cpu(), num_bands=64,
                                              max_resolution=crop_size_3d)
    traj_encoding = torch.from_numpy(traj_encoding.reshape(B*S, N-1, -1)).cuda().float()  # S, N-1, 387

    # frame patches
    frame_features = extract_frame_patches(rgbs, trajs, step=1)
    frame_features = frame_features.cuda().float()

    # generate input matrix
    input_matrix = torch.concat([traj_encoding, frame_features], axis=-1)   # S, N-1, 414

    # use perceiver io model
    pred = model(input_matrix, queries=queries)

    total_loss += criterion(pred, target_traj)

    return total_loss


def train():



    assert B == 1
    # model save path
    # model_path = 'checkpoints/01_8_64_32_1e-4_p1_avg_trajs_20:44:39.pth'  # where the ckpt is
    # state = torch.load(model_path)
    # model.load_state_dict(state['model_state'])

    # actual coeffs
    coeff_prob = 1.0

    ## autogen a name
    exp_name = 'traj_estimation'
    model_name = "%02d_%d_%d" % (B, S, N)
    lrn = "%.1e" % lr  # e.g., 5.0e-04
    lrn = lrn[0] + lrn[3:5] + lrn[-1]  # e.g., 5e-4
    model_name += "_%s" % lrn
    all_coeffs = [
        coeff_prob,
    ]
    all_prefixes = [
        "p",
    ]
    for l_, l in enumerate(all_coeffs):
        if l > 0:
            model_name += "_%s%s" % (all_prefixes[l_], utils.basic.strnum(l))
    if use_augs:
        model_name += "_A"
    model_name += "_%s" % exp_name

    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)

    ckpt_dir = 'checkpoints/%s' % model_name + '.pth'
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)
    if do_val:
        writer_v = SummaryWriter(log_dir + '/' + model_name + '/v', max_queue=10, flush_secs=60)

    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    force_double_inb = False
    force_all_inb = False

    train_dataset = flyingthingsdataset.FlyingThingsDataset(
        dset='TRAIN', subset='A',
        use_augs=use_augs,
        N=N, S=S,
        crop_size=crop_size,
        version='ac',
        force_double_inb=force_double_inb,
        force_all_inb=force_all_inb)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=B,
        shuffle=shuffle,
        num_workers=1,
        worker_init_fn=worker_init_fn,
        drop_last=True)
    train_iterloader = iter(train_dataloader)

    if cache:
        print('we will cache %d' % cache_len)
        sample_pool = utils.misc.SimplePool(cache_len, version='np')

    if do_val:
        print('not using augs in val')
        val_dataset = flyingthingsdataset.FlyingThingsDataset(dset='TEST', subset='all',
                                                              use_augs=False, N=N, S=S, crop_size=crop_size,
                                                              version='ak', force_double_inb=force_double_inb,
                                                              force_all_inb=force_all_inb)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=shuffle,
            num_workers=1)
        val_iterloader = iter(val_dataloader)

    global_step = 0

    n_pool = 100
    loss_pool_t = utils.misc.SimplePool(n_pool, version='np')
    if do_val:
        loss_pool_v = utils.misc.SimplePool(n_pool, version='np')

    total_loss = torch.tensor(0.0, requires_grad=True).to(device)
    queries = torch.ones((B, queries_dim)).to(device)
    model = PerceiverIO(depth=6, dim=dim*(2*num_band+1) + feature_dim,
                        queries_dim=queries_dim, logits_dim=logis_dim).to(device)

    criterion = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    while global_step < max_iters:

        model = model.train()

        read_start_time = time.time()
        global_step += 1

        sw_t = utils.improc.Summ_writer(
            writer=writer_t,
            global_step=global_step,
            log_freq=log_freq,
            fps=5,
            scalar_freq=int(log_freq / 2),
            just_gif=True)

        if cache:
            if (global_step) % cache_freq == 0:
                sample_pool.empty()

            if len(sample_pool) < cache_len:
                print('caching a new sample')
                try:
                    sample = next(train_iterloader)
                except StopIteration:
                    train_iterloader = iter(train_dataloader)
                    sample = next(train_iterloader)
                sample['rgbs'] = sample['rgbs'].cpu().detach().numpy()
                sample['masks'] = sample['masks'].cpu().detach().numpy()
                sample_pool.update([sample])
            else:
                sample = sample_pool.sample()
        else:
            try:
                sample = next(train_iterloader)
            except StopIteration:
                train_iterloader = iter(train_dataloader)
                sample = next(train_iterloader)
            sample['rgbs'] = sample['rgbs'].cpu().detach().numpy()
            sample['masks'] = sample['masks'].cpu().detach().numpy()

        read_time = time.time() - read_start_time
        iter_start_time = time.time()

        total_loss = run_model(model, queries, sample, criterion)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        sw_t.summ_scalar('total_loss', total_loss)
        loss_pool_t.update([total_loss.detach().cpu().numpy()])
        sw_t.summ_scalar('pooled/total_loss', loss_pool_t.mean())


        if do_val and (global_step) % val_freq == 0:
            torch.cuda.empty_cache()
            # let's do a val iter
            model.eval()
            sw_v = utils.improc.Summ_writer(
                writer=writer_v,
                global_step=global_step,
                log_freq=log_freq,
                fps=5,
                scalar_freq=int(log_freq / 2),
                just_gif=True)
            try:
                sample = next(val_iterloader)
            except StopIteration:
                val_iterloader = iter(val_dataloader)
                sample = next(val_iterloader)
            sample['rgbs'] = sample['rgbs'].cpu().detach().numpy()
            sample['masks'] = sample['masks'].cpu().detach().numpy()

            with torch.no_grad():
                total_loss = run_model(model, queries, sample, criterion, optimizer)
            sw_v.summ_scalar('total_loss', total_loss)
            loss_pool_v.update([total_loss.detach().cpu().numpy()])
            sw_v.summ_scalar('pooled/total_loss', loss_pool_v.mean())

        iter_time = time.time() - iter_start_time
        print('%s; step %06d/%d; rtime %.2f; itime %.2f; loss = %.5f' % (
            model_name, global_step, max_iters, read_time, iter_time,
            total_loss.item()))

        if not global_step % save_freq:
            torch.save(model.state_dict(), ckpt_dir)

    writer_t.close()
    if do_val:
        writer_v.close()


if __name__ == '__main__':
    # init argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_iters', type=int, help='iteration numbers',
                        default=10000)
    parser.add_argument('--use_cache', type=bool, help='whether to use cache in training;',
                        default=True)
    parser.add_argument('--cache_len', type=int, help='cache len',
                        default=None)
    parser.add_argument('--logs_dir', type=str, default='logs')
    args = parser.parse_args()

    cache = args.use_cache
    max_iters = args.max_iters
    cache_len = args.cache_len
    log_dir = args.logs_dir

    train()
