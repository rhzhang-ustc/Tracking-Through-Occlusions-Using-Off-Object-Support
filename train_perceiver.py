import time
import numpy as np
import matplotlib
import argparse
import cv2

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
from  position_encoding import generate_fourier_features

device = 'cuda'
patch_size = 8
random.seed(125)
np.random.seed(125)

## choose hyps
B = 1
S = 8
N = 64 +1 # we need to load at least 4 i think
lr = 1e-4 * 10
grad_acc = 1

crop_size = (368, 496)

max_iters = 10000
log_freq = 2
save_freq = 5000
shuffle = False
do_val = False
cache = True
cache_len = 100
cache_freq = 99999999
use_augs = False

val_freq = 10

queries_dim = 27 + 2
dim = 3
feature_dim = 27
num_band = 64
first_pred_w = 0.5
top_k_supporters = 10

log_dir = 'supporter_logs'


def write_result(frame_lst, output_path, colored):

    frame_shape = frame_lst[0].shape
    video_size = (frame_shape[1], frame_shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, 30, video_size, colored)

    for frame in frame_lst:
        video.write(frame)

    video.release()


def draw_arrows(frame, pts, dx, dy):
    pt_num = len(pts)
    for i in range(pt_num):
        pt = pts[i]
        cv2.line(frame, pt, (int(pt[0] + dx[i]), int(pt[1] + dy[i])), (0, 255, 0), 2)
        cv2.circle(frame, pt, 2, color=(255, 0, 0))
        cv2.circle(frame, (int(pt[0] + dx[i]), int(pt[1] + dy[i])), 2, color=(0, 0, 255))

    return frame


def extract_frame_patches(frames, trajs, step=1):
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
                                                    grid_x[:, s, :, i], grid_y[:, s, :, j])  # B, C, N

                output[:, s, :, :, 3*i+j] = temp.transpose(1, 2)

    # reshape, B, S, N, C*(2*step+1)*(2*step+1)
    output = output.flatten(start_dim=-2)

    return output


def run_model(model, sample, criterion, sw):

    total_loss = torch.tensor(0.0, requires_grad=True).to(device)

    rgbs = torch.from_numpy(sample['rgbs']).cuda().float()  # B, S, C, H, W
    trajs = sample['trajs'].cuda().float()  # B, S, N, 2
    valids = sample['valids'].cuda().float()  # B, S, N

    B, S, C, H, W = rgbs.shape
    _, _, N, _ = trajs.shape

    # generate target: one trajectory that needs to be estimate
    # then delete the point from trajs
    target_traj = trajs[:, :, 0:1, :]   # B, S, 1, 2
    start_target_traj = target_traj[:, 0:1, :, :]   # B, 1, 1, 2

    trajs = trajs[:, :, 1:, :]  # B, S, N-1, 2
    relative_traj = trajs - start_target_traj   # B, S, N-1, 2

    # 3d position encoding
    t_pos = torch.arange(0, S).cuda().float()
    t_pos = t_pos.repeat([B, N-1, 1]).unsqueeze(-1)  # B, N-1, S, 1
    t_pos = t_pos.transpose(1, 2)   # B, S, N-1, 1

    relative_pos = torch.concat([relative_traj, t_pos], axis=-1)  # B, S, N-1, 3

    # chop relative pos into short trajs:  B, S*(N-1), 3, 2
    # where 3, 2 means x0, y0, t0, x1, y1, t1 (start & end point)
    first_loc = relative_pos[:, :-1, :, :]  # B, S-1, N-1, 3
    second_loc = relative_pos[:, 1:, :, :]

    # 3d position encoding
    first_loc_encoding = utils.misc.get_3d_embedding(first_loc.reshape(-1, N-1, 3), num_band)
    second_loc_encoding = utils.misc.get_3d_embedding(second_loc.reshape(-1, N-1, 3), num_band)
    short_trajs_encoding = torch.concat([first_loc_encoding, second_loc_encoding], axis=-1)  # B*(S-1), N-1, 390
    short_trajs_encoding = short_trajs_encoding.reshape(B, S-1, N-1, -1)

    # frame patches
    frame_features = extract_frame_patches(rgbs, trajs, step=1)
    frame_features = frame_features.cuda().float()  # B, S, N-1, 27
    first_frame_features = frame_features[:, :-1, :, :]
    second_frame_feature = frame_features[:, 1:, :, :]

    short_trajs_features = torch.concat([first_frame_features, second_frame_feature], axis=-1)

    # generate input matrix through concat
    input_matrix = torch.concat([short_trajs_encoding, short_trajs_features], axis=-1)   # B, S-1, N-1, 444
    input_matrix = input_matrix.transpose(1, 2)
    input_matrix = input_matrix.flatten(start_dim=-2)  # B, N-1, (S-1)*444

    # use perceiver model instead of perceiver io
    # such a 'classification' model really works?
    pred = model(input_matrix)  # B, 8 * (N-1) * (S-1) ; Batch time(number of tokens) channel
    pred = pred.reshape(B, S-1, N-1, -1)

    # generate voting features
    dx0 = pred[:, :, :, 0:1]
    dx1 = pred[:, :, :, 1:2]
    dy0 = pred[:, :, :, 2:3]
    dy1 = pred[:, :, :, 3:4]
    wx0 = torch.softmax(pred[:, :, :, 4:5], dim=-2)     # B, S-1, N-1, 1
    wx1 = torch.softmax(pred[:, :, :, 5:6], dim=-2)
    wy0 = torch.softmax(pred[:, :, :, 6:7], dim=-2)
    wy1 = torch.softmax(pred[:, :, :, 7:], dim=-2)

    first_pred_traj_x = torch.sum((first_loc[:, :, :, 0:1] + dx0) * wx0, axis=-2)   # B, S-1, 1
    first_pred_traj_y = torch.sum((first_loc[:, :, :, 1:2] + dy0) * wy0, axis=-2)
    second_pred_traj_x = torch.sum((second_loc[:, :, :, 0:1] + dx1) * wx1, axis=-2)
    second_pred_traj_y = torch.sum((second_loc[:, :, :, 1:2] + dy1) * wy1, axis=-2)

    first_pred_traj = torch.concat([first_pred_traj_x, first_pred_traj_y], axis=-1)
    second_pred_traj = torch.concat([second_pred_traj_x, second_pred_traj_y], axis=-1)

    pred_traj = torch.concat([first_pred_traj[:, 0:1, :],
                              first_pred_w * first_pred_traj[:, 1:, :] + (1-first_pred_w) * second_pred_traj[:, :-1, :],
                              second_pred_traj[:, -1:, :]], axis=-2)     # B, S, 2
    pred_traj = pred_traj.unsqueeze(-2)

    # prediction vs target
    total_loss = criterion(target_traj, pred_traj)

    if sw is not None and sw.save_this:

        gt_rgb = utils.improc.preprocess_color(
            sw.summ_traj2ds_on_rgb('', target_traj[0:1], torch.mean(utils.improc.preprocess_color(rgbs[0:1]), dim=1),
                                   valids=valids[0:1], cmap='winter', only_return=True))
        gt_black = utils.improc.preprocess_color(
            sw.summ_traj2ds_on_rgb('', target_traj[0:1], torch.ones_like(rgbs[0:1, 0]) * -0.5, valids=valids[0:1],
                                   cmap='winter', only_return=True))
        sw.summ_traj2ds_on_rgb('outputs/single_trajs_on_gt_rgb', pred_traj[0:1], gt_rgb[0:1], cmap='spring')
        sw.summ_traj2ds_on_rgb('outputs/single_trajs_on_gt_black', pred_traj[0:1], gt_black[0:1], cmap='spring')

        #  test
        _, first_supporter_idx = torch.topk((wx0 ** 2 + wy0 ** 2), top_k_supporters, dim=-2)  # B, S-1, 10, 1
        frame_lst = []
        for i in range(S - 1):
            frame = rgbs[0, i]
            frame = frame.transpose(0, 2)
            frame = np.array(frame.transpose(0, 1).int().cpu())  # H, W, C
            frame = frame.copy()

            supporter_idx = first_supporter_idx[:, i, :, 0]  # B, 10
            pts = trajs[0, i]

            pts = np.array(pts[supporter_idx].int().cpu())  # B, 10, 2
            dx = np.array(dx0[0, i][supporter_idx].int().cpu())  # B, 10, 1
            dy = np.array(dy0[0, i][supporter_idx].int().cpu())
            frame_drawn = draw_arrows(frame, pts[0], dx[0], dy[0])
            frame_lst.append(np.uint8(255 * frame_drawn))

        write_result(frame_lst, "test.mp4", True)

    return total_loss


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
    model_name = model_name + '_' + model_date
    print('model_name', model_name)

    ckpt_dir = 'checkpoints/%s' % model_name + '.pth'
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

    model = Perceiver(depth=6, fourier_encode_data=False, num_classes=8 * (N-1) * (S-1),
                      num_freq_bands=64,
                      max_freq=N,
                      input_axis=1,
                      input_channels=(S-1) * 2 * ((3*num_band+3) + feature_dim),
                      final_classifier_head=True)

    model = model.cuda()
    model = torch.nn.DataParallel(model)

    criterion = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    while global_step < max_iters:

        model = model.train()

        read_start_time = time.time()
        global_step += 1

        sw_t = utils.improc.Summ_writer(
            writer=writer_t,
            global_step=global_step,
            log_freq=50,
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

        total_loss = run_model(model, sample, criterion, sw_t)

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
                total_loss = run_model(model, sample, criterion, optimizer)
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
                        default=1)
    parser.add_argument('--logs_dir', type=str, default='supporter_logs')
    args = parser.parse_args()

    cache = args.use_cache
    max_iters = args.max_iters
    cache_len = args.cache_len
    log_dir = args.logs_dir

    train()
