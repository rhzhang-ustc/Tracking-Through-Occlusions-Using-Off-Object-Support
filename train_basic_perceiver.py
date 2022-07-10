import time
import numpy as np
import matplotlib


matplotlib.use('Agg')  # suppress plot showing
import utils.py
import utils.misc
import utils.improc
import utils.grouping
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
N = 64  # we need to load at least 4 i think
lr = 1e-4
grad_acc = 1

crop_size = (368, 496)
crop_size_3d = (368, 496, S)

max_iters = 5000
log_freq = 2
save_freq = 200
shuffle = True
do_val = False
cache = False
cache_len = 101
cache_freq = 99999999
use_augs = False

val_freq = 10

queries_dim = 32
dim = 3
logis_dim = 2
num_band = 64


def extract_frame_patches(frames, trajs, step=1):
    # frames: (B, S, C, H, W)
    # trajs: (B, S, N, 2)
    # output: (B, S, N, feature_length) where feature are extracted according to (x, y)

    B, S, C, H, W = frames.size()
    span = 2 * step + 1
    feature_len = span * span * 3
    patches = torch.zeros((B, S, N, feature_len))

    # any quicker solutions?
    for b in range(B):
        for s in range(S):
            for n in range(N):

                x = int(trajs[b, s, n, 0])
                y = int(trajs[b, s, n, 1])

                x = np.clip(x, 0, H-1)
                y = np.clip(y, 0, W-1)

                try:
                    patches[b, s, n, :] = frames[b, s, :, x-step:x+step+1, y-step:y+step+1].reshape(1, -1)

                except Exception:
                    frames_new = frames[b, s, :, :, :]
                    edge_row = torch.zeros((step, W)).cuda().float().repeat(C, 1, 1) # C, step, W
                    edge_col = torch.zeros((H + 2 * step, step)).cuda().float().repeat(C, 1, 1)
                    frames_new = torch.cat([edge_row, frames_new, edge_row], axis=-2)
                    frames_new = torch.cat([edge_col, frames_new, edge_col], axis=-1)

                    patches[b, s, n, :] = frames_new[:, x:x+2*step+1, y:y+2*step+1].reshape(1, -1)

    return patches


def run_model(model, queries, sample, criterion, optimizer):

    total_loss = torch.tensor(0.0, requires_grad=True).to(device)

    rgbs = torch.from_numpy(sample['rgbs']).cuda().float()  # B, S, C, H, W
    trajs = sample['trajs'].cuda().float()  # B, S, N, 2

    B, S, C, H, W = rgbs.shape
    _, _, N, _ = trajs.shape

    target = torch.mean(trajs, axis=1).cuda().float().transpose(0, 1)    # time average, N, 1, 2

    t_pos = torch.arange(0, S).cuda().float()
    t_pos = t_pos.expand([B, N, -1]).unsqueeze(-1)
    t_pos = t_pos.transpose(1, 2)  # B, S, N, 1

    pos = torch.concat([trajs, t_pos], axis=-1)  # B, S, N, 3

    traj_encoding = generate_fourier_features(pos.reshape(-1, dim).cpu(), num_bands=64,
                                              max_resolution=crop_size_3d)
    traj_encoding = torch.from_numpy(traj_encoding.reshape(B*S, N, -1)).cuda().float()
    traj_encoding = traj_encoding.transpose(0, 1)  # N, S, 387

    # frame patches
    frame_patches = extract_frame_patches(rgbs, trajs)

    # use perceiver io model
    pred = model(traj_encoding, queries=queries)
    total_loss += criterion(pred, target)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

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
    exp_name = 'avg_trajs'
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
    log_dir = 'logs'
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
    model = PerceiverIO(depth=6, dim=dim*(2*num_band+1),
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

        total_loss = run_model(model, queries, sample, criterion, optimizer)

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
    train()
