import time
import numpy as np
import matplotlib
import argparse
import cv2
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'

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
from perceiver_pytorch import Perceiver

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import inspect
from gpu_mem_track import MemTracker

device = 'cuda'
device_ids = [0, 1, 2]
patch_size = 8
random.seed(125)
np.random.seed(125)

## choose hyps
B = 1
S = 8
N = 256 +1 # we need to load at least 4 i think
lr = 1e-5
grad_acc = 1

crop_size = (368, 496)

max_iters = 10000
log_freq = 1000
save_freq = 5000
shuffle = False
do_val = False
cache = True
cache_len = 100
cache_freq = 99999999
use_augs = False

val_freq = 10

model_depth = 1

queries_dim = 27 + 2
dim = 3
feature_dim = 27
num_band = 64
k = 10      # visualize top k supporters
feature_sample_step = 1

log_dir = 'test_logs'
video_name = "test.mp4"
model_name_suffix = 'test'
ckpt_dir = 'checkpoints_test'

# model_path = 'checkpoints/01_8_257_1e-4_p1_traj_estimation_00:52:54_cache_len_15.pth'  # where the ckpt is
use_ckpt = False


def write_result(frame_lst, output_path, colored):

    frame_shape = frame_lst[0].shape
    video_size = (frame_shape[1], frame_shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, 30, video_size, colored)

    for frame in frame_lst:
        video.write(frame)

    video.release()


def draw_arrows(frame, supporters, votes, groundtruth, pred, order=False):
    # order: draw arrow & suggest its weights
    # N, 3: supporters / votes
    # 1, 2: groundtruth /pred

    k, _ = supporters.shape
    H, W, C = frame.shape
    frame_gray = (0.299 * frame[:, :, 0]) + (0.587 * frame[:, :, 1]) + (0.114 * frame[:, :, 2])
    frame_gray = frame_gray.reshape(H, W, 1).repeat(3, axis=-1)

    frame = (frame_gray + frame)/2

    for idx in range(k):
        pt = supporters[idx]
        dxy = votes[idx]

        start_x = int(pt[0])
        start_y = int(pt[1])
        end_x = int(pt[0] + dxy[0])
        end_y = int(pt[1] + dxy[1])

        start_pt = (start_x, start_y)
        end_pt = (end_x, end_y)

        cv2.line(frame, start_pt, end_pt, (255, 0, 0))
        # cv2.rectangle(frame, (start_pt[0] - 2, start_pt[1] - 2), (start_pt[0] + 2, start_pt[1] + 2), (0, 0, 255))
        cv2.rectangle(frame, (end_pt[0] - 1, end_pt[1] - 1), (end_pt[0] + 1, end_pt[1] + 1), (255, 0, 0))

    cv2.circle(frame, (int(groundtruth[0]), int(groundtruth[1])), 4, (0, 0, 255), thickness=-1)
    cv2.circle(frame, (int(pred[0]), int(pred[1])), 4, (255, 0, 0), thickness=-1)

    return frame


def extract_frame_patches(frames, trajs, step=feature_sample_step):
    # steps: dilate conv style
    # frames: (B, S, C, H, W)
    # trajs: (B, S, N, 2)
    # output: (B, S, N, feature_length) where feature are extracted according to (x, y)

    _, _, C, H, W = frames.shape
    B, S, N, _ = trajs.shape

    # generate grid matrix for flow, B, S, N*9, 2
    x = trajs[:, :, :, 0:1].clone().detach().int()  # B, S, N, 1
    y = trajs[:, :, :, 1:].clone().detach().int()

    x_range = torch.concat([x-step, x, x+step], axis=-1).unsqueeze(-1)  # B, S, N, 3, 1
    y_range = torch.concat([y-step, y, y+step], axis=-1).unsqueeze(-2)  # B, S, N, 1, 3

    grid_x = x_range.repeat(1, 1, 1, 1, 3).flatten(start_dim=-2)  # meshgrid, B, S, N, 9
    grid_y = y_range.repeat(1, 1, 1, 3, 1).flatten(start_dim=-2)

    # bilinear sample, B, S, N, C, 9
    output = torch.zeros((B, S, N, C, 9))
    for s in range(S):
        for i in range(3):
            for j in range(3):
                temp = utils.samp.bilinear_sample2d(frames[:, s],
                                                    grid_x[:, s, :, i],
                                                    grid_y[:, s, :, j])  # B, C, N

                output[:, s, :, :, 3*i+j] = temp.transpose(1, 2)

    # reshape, B, S, N, C*(2*step+1)*(2*step+1)
    output = output.flatten(start_dim=-2)

    return output


def run_model(model, sample, criterion, sw):

    total_loss = torch.tensor(0.0, requires_grad=True, device=device)

    rgbs = torch.from_numpy(sample['rgbs']).cuda().float()  # B, S, C, H, W
    trajs = sample['trajs'].cuda().float()  # B, S, N, 2
    valids = sample['valids'].cuda().float()  # B, S, N

    B, S, C, H, W = rgbs.shape
    _, _, N, _ = trajs.shape

    '''
    generate target: one trajectory that needs to be estimate, then delete the point from trajs
    '''
    target_traj = trajs[:, :, 0:1, :]   # B, S, 1, 2
    start_target_traj = target_traj[:, 0:1, :, :]   # B, 1, 1, 2

    trajs = trajs[:, :, 1:, :]  # B, S, N-1, 2
    relative_traj = trajs - start_target_traj   # B, S, N-1, 2

    # 3d position encoding
    t_pos = torch.arange(0, S).cuda().float()
    t_pos = t_pos.repeat([B, N-1, 1]).unsqueeze(-1)  # B, N-1, S, 1
    t_pos = t_pos.transpose(1, 2)   # B, S, N-1, 1

    relative_pos = torch.concat([relative_traj, t_pos], axis=-1)  # B, S, N-1, 3

    '''
    chop relative pos into short trajs:  B, S*(N-1), 3, 2
    where 3, 2 means x0, y0, t0, x1, y1, t1 (start & end point)
    '''
    start_loc = relative_pos[:, :-1, :, :].reshape(B, -1, 3)  # B, (S-1)*(N-1), 3
    end_loc = relative_pos[:, 1:, :, :].reshape(B, -1, 3)

    # parameter M indicate the number of trajs
    true_traj_mask = (0 < start_loc[:, :, 0]) & (start_loc[:, :, 0] < W-1) \
                     & (0 < end_loc[:, :, 0]) & (end_loc[:, :, 0] < W - 1) \
                     & (0 < start_loc[:, :, 1]) & (start_loc[:, :, 1] < H - 1) \
                     & (0 < end_loc[:, :, 1]) & (end_loc[:, :, 1] < H - 1)

    true_traj_mask = true_traj_mask.unsqueeze(-1)   # 1, M, 1
    M = int(true_traj_mask.sum())

    if M <= 0:
        print('false data')
        return total_loss

    '''
    filter the data
    '''
    start_loc = torch.masked_select(start_loc, true_traj_mask.repeat(1, 1, 3)).reshape(M, B, -1)  # M, 1, 3
    end_loc = torch.masked_select(end_loc, true_traj_mask.repeat(1, 1, 3)).reshape(M, B, -1)

    '''
    3d position encoding
    '''
    start_loc_encoding = utils.misc.get_3d_embedding(start_loc, num_band)
    end_loc_encoding = utils.misc.get_3d_embedding(end_loc, num_band)
    short_trajs_encoding = torch.concat([start_loc_encoding, end_loc_encoding], axis=-1)  # M, 1, 390

    '''
    frame patches
    '''
    frame_features = extract_frame_patches(rgbs, trajs, step=1).cuda().float()  # B, S-1, N-1, 27
    start_frame_features = frame_features[:, :-1, :, :]
    end_frame_feature = frame_features[:, 1:, :, :]

    short_trajs_features = torch.concat([start_frame_features, end_frame_feature], axis=-1).reshape(B, -1, 2*feature_dim)
    short_trajs_features = torch.masked_select(short_trajs_features, true_traj_mask.repeat(1, 1, 2*feature_dim)).reshape(M, B, -1)

    '''
    relative motion
    '''
    target_motion = (target_traj[:, 1:2, :, :] - start_target_traj)[0]   # 1, 1, 2
    target_motion = torch.concat([target_motion, torch.ones(1, 1, 1).cuda().float()], dim=-1)  # 1, 1, 3
    motion = end_loc - start_loc    # M, 1, 3
    relative_motion = motion - target_motion

    '''
    use perceiver model instead of perceiver io
    '''
    input_matrix = torch.concat([short_trajs_encoding, short_trajs_features, relative_motion], axis=-1)  # M, 1, 447
    input_matrix = input_matrix.to(device)

    pred = model(input_matrix)  # M, 6
    pred = pred.reshape(-1, 1, 6)   # M, 1, 6

    '''
    generate voting features M, 1, 6
    '''
    dx0 = pred[:, :, 0:1]
    dy0 = pred[:, :, 1:2]
    dx1 = pred[:, :, 2:3]
    dy1 = pred[:, :, 3:4]
    dt = torch.zeros(dx0.shape).cuda().float()

    w0 = pred[:, :, 4:5]  # M, 1, 1  weight for vote that begins at start point
    w1 = pred[:, :, 5:6]

    # make sure that w are normalized inside the same timestep, generate prediction

    start_vote_matrix = torch.concat([dx0, dy0, dt], dim=-1)    # M, 1, 3
    end_vote_matrix = torch.concat([dx1, dy1, dt], dim=-1)

    frame_lst = []  # for visualization
    pred_traj = torch.zeros(B, S, 1, 2)

    # separate trajs that belongs to different time step
    for i in range(S):

        idx_start = start_loc[:, :, -1] == i
        idx_end = end_loc[:, :, -1] == i

        if torch.sum(idx_start) + torch.sum(idx_end) == 0:
            continue

        supporters_start_loc = start_loc[idx_start]
        supporters_end_loc = end_loc[idx_end]
        supporters = torch.concat([supporters_start_loc, supporters_end_loc], dim=0)    # M, 3

        votes = torch.concat([start_vote_matrix[idx_start], end_vote_matrix[idx_end]], dim=0)
        w = torch.concat([w0[idx_start], w1[idx_end]], dim=0)

        vote_pts = supporters + votes
        norm_w = torch.softmax(w, dim=0)

        pred_traj_i = torch.sum(norm_w * vote_pts, dim=0)[None, 0:2]
        pred_traj[:, i, :, :] = pred_traj_i

        target_traj_i = target_traj[0, i]   # 1, 2

        total_loss = total_loss + criterion(pred_traj_i, target_traj_i)

        '''
        additional loss: force topk points to have reasonable result
        '''
        try:
            k_temp = k
            _, top_k_index = torch.topk(norm_w, k_temp, dim=0)  # M, 1
        except Exception:
            k_temp = len(norm_w)
            _, top_k_index = torch.topk(norm_w, k_temp, dim=0)

        k_supporter = supporters.index_select(0, top_k_index[:, 0])  # M, 3
        k_votes = votes.index_select(0, top_k_index[:, 0])  # M , 3

        pred_matrix = (k_supporter + k_votes)[:, 0:2]
        total_loss = total_loss + criterion(pred_matrix, target_traj_i.repeat(k_temp, 1))

        '''
        visualization
        '''
        if sw is not None and sw.save_this:
            frame = rgbs[0, i].transpose(0, 2).transpose(0, 1).cpu()
            frame = np.array(frame)[:, :, ::-1]
            frame = np.ascontiguousarray(frame, dtype=np.int32)    # H, W, C

            frame_drawn = draw_arrows(frame, k_supporter.cpu().detach(),
                                      k_votes.cpu().detach(),
                                      target_traj_i[0],
                                      pred_traj_i[0],
                                      order=False)

            frame_lst.append(np.array(frame_drawn, dtype=np.uint8))

    if sw is not None and sw.save_this:

        gt_rgb = utils.improc.preprocess_color(
            sw.summ_traj2ds_on_rgb('', target_traj[0:1], torch.mean(utils.improc.preprocess_color(rgbs[0:1]), dim=1),
                                   valids=valids[0:1], cmap='winter', only_return=True))
        gt_black = utils.improc.preprocess_color(
            sw.summ_traj2ds_on_rgb('', target_traj[0:1], torch.ones_like(rgbs[0:1, 0]) * -0.5, valids=valids[0:1],
                                   cmap='winter', only_return=True))
        sw.summ_traj2ds_on_rgb('outputs/single_trajs_on_gt_rgb', pred_traj[0:1], gt_rgb[0:1], cmap='spring')
        sw.summ_traj2ds_on_rgb('outputs/single_trajs_on_gt_black', pred_traj[0:1], gt_black[0:1], cmap='spring')

        sw.summ_rgbs('inputs_0/rgbs', utils.improc.preprocess_color(rgbs[0:1]).unbind(1))
        if len(frame_lst):
            write_result(frame_lst, video_name, True)

    return total_loss/S


def train():

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
    model_name = model_name + '_' + model_date + '_' + model_name_suffix
    print('model_name', model_name)

    ckpt_path = ckpt_dir + '/' + model_name + '.pth'
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)
    if do_val:
        writer_v = SummaryWriter(log_dir + '/' + model_name + '/v', max_queue=10, flush_secs=60)

    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    train_dataset = flyingthingsdataset.FlyingThingsDataset(
        dset='TRAIN', subset='A',
        use_augs=use_augs,
        N=N, S=S,
        crop_size=crop_size)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=B,
        shuffle=shuffle,
        num_workers=0,
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
                                                              version='ak')
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

    # model = PerceiverIO(depth=6, dim=S * (dim*(2*num_band+1) + feature_dim),
                        # queries_dim=queries_dim, logits_dim=logis_dim).to(device)

    model = Perceiver(depth=model_depth, fourier_encode_data=False, num_classes=6,
                      num_freq_bands=64,
                      max_freq=N,
                      input_axis=1,
                      num_latents=512,
                      latent_dim=512,
                      input_channels=2 * ((3*num_band+3) + feature_dim) + 3,
                      final_classifier_head=True)

    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)

    if use_ckpt:
        state = torch.load(model_path)
        model.load_state_dict(state, strict=False)

    criterion = nn.L1Loss()
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    scaler = torch.cuda.amp.GradScaler()

    while global_step < max_iters:

        model = model.train()

        read_start_time = time.time()
        global_step += 1

        sw_t = utils.improc.Summ_writer(
            writer=writer_t,
            global_step=global_step,
            log_freq=log_freq,
            fps=5,
            scalar_freq=2,
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

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            total_loss = run_model(model, sample, criterion, sw_t)

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
                scalar_freq=2,
                just_gif=True)
            try:
                sample = next(val_iterloader)
            except StopIteration:
                val_iterloader = iter(val_dataloader)
                sample = next(val_iterloader)
            sample['rgbs'] = sample['rgbs'].cpu().detach().numpy()
            sample['masks'] = sample['masks'].cpu().detach().numpy()

            with torch.no_grad():
                total_loss = run_model(model, sample, criterion, optimizer)
            sw_v.summ_scalar('total_loss', total_loss)
            loss_pool_v.update([total_loss.detach().cpu().numpy()])
            sw_v.summ_scalar('pooled/total_loss', loss_pool_v.mean())

        iter_time = time.time() - iter_start_time
        print('%s; step %06d/%d; rtime %.2f; itime %.2f; loss = %.5f' % (
            model_name, global_step, max_iters, read_time, iter_time,
            total_loss.item()))

        if not global_step % save_freq:
            torch.save(model.state_dict(), ckpt_path)

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
                        default=1)

    args = parser.parse_args()

    cache = args.use_cache
    max_iters = args.max_iters
    cache_len = args.cache_len


    train()
