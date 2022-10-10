import time
import numpy as np
import matplotlib
import argparse
import cv2
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

matplotlib.use('Agg')  # suppress plot showing
import utils.py
import utils.misc
import utils.improc
import utils.grouping
import utils.samp
import utils.basic
import random
from PIL import Image

# import datasets
import flyingthingsdataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import saverloader

from tensorboardX import SummaryWriter

from net.perceiver_pytorch import MLP
from net.pips import BasicEncoder, Pips

device = 'cuda'
device_ids = [0]
patch_size = 8
random.seed(125)
np.random.seed(125)

## choose hyps
B = 1
S = 8
S_interval = 4
S_out = 16

N = 16 + 1  # we need to load at least 4 i think
lr = 1e-4
grad_acc = 1

crop_size = (368, 496)

max_iters = 10000
log_freq = 1000
save_freq = 5000
shuffle = True
do_val = False

iters = 2

cache = True
cache_len = 10
cache_freq = 99999999
use_augs = True

val_freq = 100

model_depth = 1

queries_dim = 27 + 2
dim = 3
feature_map_dim = 128
encoder_stride = 8

num_band = 32
k = 10  # supervision
k_vis = 100  # visualize
feature_sample_step = 1
vis_threshold = 0.1

beta = 1 # importance score of top k loss
alpha = 1

init_dir = 'reference_model'

log_dir = 'test_logs'
model_name_suffix = 'mutual_16_iter2_cache'
ckpt_dir = 'checkpoints_mutual_16_iter2_cache'
num_worker = 12

use_ckpt = False


def vis_vote(frame, supporters, votes, weights, vis_thres, groundtruth, pred, color=False):
    # order: draw arrow & suggest its weights
    # N, 3: supporters / votes
    # 1, 2: groundtruth /pred

    k, _ = supporters.shape
    H, W, C = frame.shape

    if not color:
        frame = np.mean(frame, axis=-1)
        frame = np.expand_dims(frame, 2)
        frame = np.repeat(frame, 3, 2)

    for idx in range(k):
        w = weights[idx]
        if w < vis_thres:
            continue
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
        cv2.putText(frame, str(w.item()), (start_pt[0] - 2, start_pt[1] - 2), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

    cv2.circle(frame, (int(groundtruth[0]), int(groundtruth[1])), 3, (0, 0, 255), thickness=-1)
    cv2.circle(frame, (int(pred[0]), int(pred[1])), 3, (255, 0, 0), thickness=-1)

    return frame


def run_pips(model, rgbs, target, N, sw):
    rgbs = rgbs.cuda().float()  # B, S, C, H, W

    B, S, C, H, W = rgbs.shape
    rgbs_ = rgbs.reshape(B * S, C, H, W)
    H_, W_ = crop_size #360, 640
    rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
    H, W = H_, W_
    rgbs = rgbs_.reshape(B, S, C, H, W)

    # pick N points to track; we'll use a uniform grid
    N_ = np.sqrt(N).round().astype(np.int32)
    grid_y, grid_x = utils.basic.meshgrid2d(B, N_, N_, stack=False, norm=False, device='cuda')
    grid_y = 8 + grid_y.reshape(B, -1) / float(N_ - 1) * (H - 16)
    grid_x = 8 + grid_x.reshape(B, -1) / float(N_ - 1) * (W - 16)
    xy = torch.stack([grid_x, grid_y], dim=-1)  # B, N_*N_, 2
    _, S, C, H, W = rgbs.shape

    xy = torch.concat([xy, target[:, 0, :, :]], dim=1)   # B, N_*N_ + 1, 2

    preds, preds_anim, vis_e, stats = model(xy, rgbs, iters=6)
    trajs_e = preds[-1]

    target_e = trajs_e[:, :, -1:, :]    # B, S, 1, 2
    trajs_e = trajs_e[:, :, :-1, :]
    vis_e = vis_e[:, :, :-1]

    pad = 50
    rgbs = F.pad(rgbs.reshape(B * S, 3, H, W), (pad, pad, pad, pad), 'constant', 0).reshape(B, S, 3, H + pad * 2,
                                                                                            W + pad * 2)
    trajs_e = trajs_e + pad

    if sw is not None and sw.save_this:
        linewidth = 2

        # visualize the input
        o1 = sw.summ_rgbs('inputs/rgbs', utils.improc.preprocess_color(rgbs[0:1]).unbind(1), only_return=True)
        # visualize the trajs overlaid on the rgbs
        o2 = sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e[0:1], utils.improc.preprocess_color(rgbs[0:1]),
                                     cmap='spring', linewidth=linewidth, only_return=True)
        # visualize the trajs alone
        o3 = sw.summ_traj2ds_on_rgbs('outputs/trajs_on_black', trajs_e[0:1], torch.ones_like(rgbs[0:1]) * -0.5,
                                     cmap='spring', linewidth=linewidth, only_return=True)
        # concat these for a synced wide vis
        wide_cat = torch.cat([o1, o2, o3], dim=-1)
        sw.summ_rgbs('outputs/wide_cat', wide_cat.unbind(1))

    return target_e, trajs_e - pad, vis_e


def get_mutual_support(rgbs, target_traj, trajs, valids, model, encoder):

    # use trajs to support target_traj
    # trajs: B, S, N0, 2
    # target_traj: B, S, N0, 2

    _, _, N0, _ = trajs.shape
    B, S, C, H, W = rgbs.shape

    start_target_traj = target_traj[:, 0:1, :, :]  # B, 1, N0, 2
    relative_traj = trajs - start_target_traj  # B, S, N0, 2

    # 3d position encoding
    t_pos = torch.arange(0, S).cuda().float()
    t_pos = t_pos.repeat([B, N0, 1]).unsqueeze(-1)  # B, N0, S, 1
    t_pos = t_pos.transpose(1, 2)  # B, S, N0, 1

    relative_pos = torch.concat([relative_traj, t_pos], axis=-1)  # B, S, N0, 3
    pos = torch.concat([trajs, t_pos], axis=-1)

    trajs_encoding = utils.misc.get_3d_embedding(relative_pos.reshape(B * S, N0, 3), num_band)
    trajs_encoding = trajs_encoding.reshape(B, S, N0, -1)  # B, S, N-1, 99
    trajs_encoding = trajs_encoding.permute(0, 2, 1, 3).reshape(B, N0, -1)  # B, N0, S*99

    frame_features_map = encoder(rgbs[:, 0].reshape(B, C, H, W)).reshape(B, 1, feature_map_dim,
                                                                         H // encoder_stride,
                                                                         W // encoder_stride)  # 1, 8, 128, 46, 62

    trajs_features = utils.samp.bilinear_sample2d(frame_features_map[:, 0, :],
                                                  pos[:, 0, :, 0] / encoder_stride,
                                                  pos[:, 0, :, 1] / encoder_stride)  # B, 128, N0
    trajs_features = trajs_features.permute(0, 2, 1)  # B, N0, 128

    target_traj_feature = utils.samp.bilinear_sample2d(frame_features_map[:, 0, :],
                                                       start_target_traj[:, 0, :, 0] / encoder_stride,
                                                       start_target_traj[:, 0, :, 1] / encoder_stride)  # B, 128, 1
    target_traj_feature = target_traj_feature.permute(0, 2, 1)  # B, N0, 128

    target_motion = (target_traj[:, 1:2, :, :] - start_target_traj)  # B, 1, N0, 2
    target_motion = torch.concat([target_motion, torch.ones(B, 1, N0, 1).cuda().float()], dim=-1)  # B, 1, 1, 3
    motion = pos[:, 1:, :, :] - pos[:, :-1, :, :]  # B, S-1, N-1, 3

    relative_motion = motion - target_motion

    relative_motion = relative_motion[:, :, :, 0:2]

    relative_motion_encoding = utils.misc.get_2d_embedding(relative_motion.reshape(B * (S - 1), N0, 2), num_band)
    relative_motion_encoding = relative_motion_encoding.reshape(B, S - 1, N0, -1)  # B, S-1, N0, 66
    relative_motion_encoding = relative_motion_encoding.permute(0, 2, 1, 3).reshape(B, N0, -1)  # B, N0, 66*(S-1)

    # target feature
    input_matrix = torch.concat([trajs_encoding,
                                 trajs_features,
                                 relative_motion_encoding,
                                 target_traj_feature,
                                 valids.permute(0, 2, 1)], axis=-1)

    input_matrix = input_matrix.to(device)  # B, N0, 1518
    pred = model(input_matrix)  # B, N0, S*3
    pred = pred.reshape(B, N0, S, 3).permute(0, 2, 1, 3)    # B, S, N0, 3

    w_supporter_target = pred[0, 0, :, 2]

    return w_supporter_target


def select_supporters(model, encoder, rgbs, trajs, target_traj, valids):

    _, _, N0, _ = trajs.shape

    '''
    generate W_supporters_target
    '''
    w_supporter_target = get_mutual_support(rgbs, target_traj.repeat(1, 1, N0, 1), trajs, valids, model, encoder)
    w_target_supporter = get_mutual_support(rgbs, trajs, target_traj.repeat(1, 1, N0, 1), valids, model, encoder)

    mutual_loss = torch.abs(w_target_supporter - w_supporter_target).mean()

    return mutual_loss


def run_model(model, encoder, timestep,
              first_frame, start_target_traj, target_motion,
              rgbs, trajs, target_traj, curr_e_traj, valids, criterion, sw):
    # first_frame: B, C, H, W

    total_loss = torch.tensor(0.0, requires_grad=True, device=device)

    B, S, C, H, W = rgbs.shape
    _, _, N0, _ = trajs.shape

    '''
    generate target: one trajectory that needs to be estimate
    '''
    relative_traj = trajs - curr_e_traj  # B, S, N0, 2

    # 3d position encoding
    t_pos = torch.tensor(timestep).cuda().float()
    t_pos = t_pos.repeat([B, N0, 1]).unsqueeze(-1)  # B, N0, S, 1
    t_pos = t_pos.transpose(1, 2)  # B, S, N0, 1

    relative_pos = torch.concat([relative_traj, t_pos], axis=-1)  # B, S, N0, 3
    pos = torch.concat([trajs, t_pos], axis=-1)

    '''
    3d position encoding
    '''

    trajs_encoding = utils.misc.get_3d_embedding(relative_pos.reshape(B * S, N0, 3), num_band)
    trajs_encoding = trajs_encoding.reshape(B, S, N0, -1)  # B, S, N-1, 99

    trajs_encoding = trajs_encoding.permute(0, 2, 1, 3).reshape(B, N0, -1)  # B, N0, S*99

    '''
    Sample from CNN features
    '''
    first_frame_features_map = encoder(first_frame.reshape(B, C, H, W)).reshape(B, 1, feature_map_dim,
                                                                       H // encoder_stride,
                                                                       W // encoder_stride)  # 1, 8, 128, 46, 62

    frame_features_map = encoder(rgbs[:, 0, :, :, :].reshape(B, C, H, W)).reshape(B, 1, feature_map_dim,
                                                                       H // encoder_stride,
                                                                       W // encoder_stride)  # 1, 8, 128, 46, 62

    trajs_features = utils.samp.bilinear_sample2d(frame_features_map[:, 0, :],
                                                  pos[:, 0, :, 0] / encoder_stride,
                                                  pos[:, 0, :, 1] / encoder_stride)  # B, 128, N0

    trajs_features = trajs_features.permute(0, 2, 1)  # B, N0, 128

    '''
    target vector
    '''

    target_traj_feature = utils.samp.bilinear_sample2d(first_frame_features_map[:, 0, :],
                                                       start_target_traj[:, 0, 0, 0:1] / encoder_stride,
                                                       start_target_traj[:, 0, 0, 1:2] / encoder_stride)  # B, 128, 1

    target_traj_feature = target_traj_feature.permute(0, 2, 1).repeat(1, N0, 1)  # B, N0, 128


    '''
    relative motion
    '''

    # target_motion = (target_traj[:, 1:2, :, :] - start_target_traj)  # B, 1, 1, 2
    target_motion = torch.concat([target_motion, torch.ones(B, 1, 1, 1).cuda().float()], dim=-1)  # B, 1, 1, 3
    motion = pos[:, 1:, :, :] - pos[:, :-1, :, :]  # B, S-1, N-1, 3

    relative_motion = motion - target_motion

    relative_motion = relative_motion[:, :, :, 0:2]

    relative_motion_encoding = utils.misc.get_2d_embedding(relative_motion.reshape(B * (S-1), N0, 2), num_band)
    relative_motion_encoding = relative_motion_encoding.reshape(B, S-1, N0, -1)  # B, S-1, N0, 66
    relative_motion_encoding = relative_motion_encoding.permute(0, 2, 1, 3).reshape(B, N0, -1)  # B, N0, 66*(S-1)

    '''
    use perceiver model instead of perceiver io
    '''

    # target feature
    input_matrix = torch.concat([trajs_encoding,
                                 trajs_features,
                                 relative_motion_encoding,
                                 target_traj_feature,
                                 valids.permute(0, 2, 1)], axis=-1)

    input_matrix = input_matrix.to(device)  # B, N0, 1518
    pred = model(input_matrix)  # B, N0, S*3

    pred = pred.reshape(B, N0, S, -1).permute(0, 2, 1, 3)   # B, S, N0, 3

    '''
    generate voting features
    '''
    dx = pred[:, :, :, 0:1]  # B, S, N0, 1
    dy = pred[:, :, :, 1:2]
    dt = torch.zeros(dx.shape).cuda().float()
    ws = pred[:, :, :, 2:3]  # B, S, N0, 1  weight for vote that begins at start point

    # make sure that w are normalized inside the same timestep, generate prediction

    vote_matrix = torch.concat([dx, dy, dt], dim=-1)  # B, S, N0, 3

    pred_traj = torch.zeros(B, S, 1, 2).cuda()

    for i in range(S):

        supporters = pos[:, i:i + 1, :, :]  # B, 1, N0, 3
        votes = vote_matrix[:, i:i + 1, :, :]  # B, 1, N0, 3
        w = ws[:, i:i + 1, :, :]  # B, 1, N0, 1

        vote_pts = supporters + votes
        norm_w = torch.softmax(w, dim=2)  # B, 1, N0, 1

        pred_traj_i = torch.sum(norm_w * vote_pts, dim=2)[:, :, 0:2]  # B, 1, 2
        pred_traj[:, i, :, :] = pred_traj_i

        target_traj_i = target_traj[:, i, :, :]  # B, 1, 2

        total_loss = total_loss + criterion(pred_traj_i, target_traj_i)

        '''
        additional loss: force topk points to have reasonable result
        '''
        try:
            k_temp = k_vis
            _, top_k_index = torch.topk(norm_w, k_temp, dim=2)
        except Exception:
            k_temp = len(norm_w)
            _, top_k_index = torch.topk(norm_w, k_temp, dim=2)  # B, 1, 100, 1

        k_supporter = supporters.index_select(2, top_k_index[0, 0, :, 0])  # 1, 1, 100, 3
        k_votes = votes.index_select(2, top_k_index[0, 0, :, 0])
        k_w = norm_w.index_select(2, top_k_index[0, 0, :, 0])

        pred_matrix = (k_supporter + k_votes)[:, :, :, 0:2]  # 1, 1, 100, 2

        # more easy way
        if k > k_temp:
            k_i = k_temp  # number of supporters that recieve loss
        else:
            k_i = k

        total_loss = total_loss + beta * criterion(pred_matrix[:, :, :k_i, :],
                                                   target_traj_i[:, None, :, :].repeat(1, 1, k_i, 1))  # loss for k


    avg_traj_error = torch.mean(torch.norm(pred_traj-target_traj, 2, dim=-1))

    return total_loss / S, \
           avg_traj_error, \
           pred_traj


def compute_delta_avg(pred_traj, trajs, delta=1):
    # pred_trajs: B, S, N0, 2
    # target_traj: B, S, N0, 2

    B, S, N0, _ = trajs.shape
    valid_index = torch.sum(torch.abs(pred_traj-trajs), dim=-1) < delta     # B, S, N0

    return torch.sum(valid_index)/(B * S * N0)



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

    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)
    if do_val:
        writer_v = SummaryWriter(log_dir + '/' + model_name + '/v', max_queue=10, flush_secs=60)

    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    train_dataset = flyingthingsdataset.FlyingThingsDataset(
        dset='TRAIN', subset='all',
        use_augs=use_augs,
        N=N, S=S,
        S_out=S_out,
        zigzag=True,
        crop_size=crop_size)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=B,
        shuffle=shuffle,
        num_workers=num_worker,
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
                                                              )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=shuffle,
            num_workers=12,
            pin_memory= True)
        val_iterloader = iter(val_dataloader)

    global_step = 0

    n_pool = 100
    loss_pool_0_t = utils.misc.SimplePool(n_pool, version='np')
    loss_pool_4_t = utils.misc.SimplePool(n_pool, version='np')
    loss_pool_8_t = utils.misc.SimplePool(n_pool, version='np')

    avg_error_pool_0_t = utils.misc.SimplePool(n_pool, version='np')
    avg_error_pool_4_t = utils.misc.SimplePool(n_pool, version='np')
    avg_error_pool_8_t = utils.misc.SimplePool(n_pool, version='np')

    model_loss_pool_0_t = utils.misc.SimplePool(n_pool, version='np')
    model_loss_pool_4_t = utils.misc.SimplePool(n_pool, version='np')
    model_loss_pool_8_t = utils.misc.SimplePool(n_pool, version='np')

    mutual_loss_pool_0_t = utils.misc.SimplePool(n_pool, version='np')
    mutual_loss_pool_4_t = utils.misc.SimplePool(n_pool, version='np')
    mutual_loss_pool_8_t = utils.misc.SimplePool(n_pool, version='np')


    delta_avg_1_pool_t = utils.misc.SimplePool(n_pool, version='np')
    delta_avg_4_pool_t = utils.misc.SimplePool(n_pool, version='np')
    delta_avg_16_pool_t = utils.misc.SimplePool(n_pool, version='np')

    n_pool_v = 100
    if do_val:
        loss_pool_v = utils.misc.SimplePool(n_pool_v, version='np')
        avg_error_pool_v = utils.misc.SimplePool(n_pool, version='np')

    model = MLP(in_dim=S * (3 * num_band + 3) + 2 * feature_map_dim + (S-1) * (2 * num_band + 2) + S,
                out_dim=S * 3,
                hidden_dim=4096
    )

    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)

    encoder = BasicEncoder(input_dim=3, output_dim=feature_map_dim, stride=encoder_stride)
    encoder = encoder.to(device)
    encoder = torch.nn.DataParallel(encoder, device_ids=device_ids)

    pips = Pips(stride=4).cuda()
    saverloader.load(init_dir, pips)
    pips.eval()

    if use_ckpt:

        saverloader.load(ckpt_dir, model, optimizer=None, scheduler=None, model_ema=None, step=0, model_name='model',
                         ignore_load=None)
        saverloader.load(ckpt_dir, encoder, optimizer=None, scheduler=None, model_ema=None, step=0, model_name='encoder',
                         ignore_load=None)

    criterion = nn.L1Loss()
    optimizer = optim.AdamW([
        {'params': model.parameters(), 'lr': lr},
        {'params': encoder.parameters(), 'lr': lr}
    ])

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

        rgbs = torch.from_numpy(sample['rgbs']).cuda().float()  # B, S, C, H, W

        # ground truth trajs
        # trajs = sample['trajs'].cuda().float()  # B, S, N, 2

        valids = sample['valids'].cuda().float()  # B, S, N
        rgbs = torch.from_numpy(sample['rgbs']).cuda().float()  # B, S, C, H, W
        trajs = sample['trajs'].cuda().float()  # B, S, N, 2

        '''
        train process
        '''

        read_time = time.time() - read_start_time
        iter_start_time = time.time()

        optimizer.zero_grad()

        N0 = trajs.shape[2]
        target_idx = int(torch.randint(0, N0 - 1, (1,)))
        target_traj = trajs[:, :, target_idx:target_idx+1, :]  # B, S, 1, 2

        pred_traj = torch.zeros(B, S_out, 1, 2).cuda().float()
        total_loss = torch.tensor(0.0, requires_grad=True).to(device)
        model_loss = torch.tensor(0.0, requires_grad=True).to(device)
        mutual_loss = torch.tensor(0.0, requires_grad=True).to(device)

        curr_e_traj = target_traj[:, 0:1, :, :].repeat(1, S_out, 1, 1)

        for _ in range(iters):

            for i in range(S_out // 4 - 1):

                frame_index = range(4*i, 4*i+8)
                rgbs_sub = rgbs[:, frame_index, :, :, :]
                target_traj_sub = target_traj[:, frame_index, :, :]
                curr_e_traj_sub = curr_e_traj[:, frame_index, :, :]

                # with torch.no_grad():
                target_e, trajs_e, vis_e = run_pips(pips, rgbs_sub, target_traj_sub, N, sw_t)  # N-2

                mutual_loss_sub = select_supporters(model, encoder, rgbs_sub, trajs_e, target_e, vis_e)

                model_loss_sub, ate, pred_traj_sub = run_model(model, encoder, frame_index,
                                                           rgbs[:, 0, :, :, :], target_traj[:, 0:1, :, :],
                                                           target_traj[:, 1:2, :, :] - target_traj[:, 0:1, :, :],
                                                           rgbs_sub, trajs_e, target_traj_sub,
                                                           curr_e_traj_sub,
                                                           vis_e, criterion, sw_t)

                model_loss = model_loss_sub + model_loss
                mutual_loss = mutual_loss_sub + mutual_loss

                pred_traj[:, frame_index, :, :] = pred_traj_sub

                if i == 0 and _ == 0:

                    sw_t.summ_scalar('unpooled/model_loss_' + str(i), model_loss)
                    model_loss_pool_0_t.update([model_loss.detach().cpu().numpy()])
                    sw_t.summ_scalar('pooled/model_loss_' + str(i), model_loss_pool_0_t.mean())

                    sw_t.summ_scalar('unpooled/mutual_loss_' + str(i), mutual_loss)
                    mutual_loss_pool_0_t.update([mutual_loss.detach().cpu().numpy()])
                    sw_t.summ_scalar('pooled/mutual_loss_' + str(i), mutual_loss_pool_0_t.mean())

                    sw_t.summ_scalar('unpooled/total_loss_' + str(i), total_loss)
                    loss_pool_0_t.update([total_loss.detach().cpu().numpy()])
                    sw_t.summ_scalar('pooled/total_loss_' + str(i), loss_pool_0_t.mean())

                    sw_t.summ_scalar('unpooled/average_error_' + str(i), ate)
                    avg_error_pool_0_t.update([ate.detach().cpu().numpy()])
                    sw_t.summ_scalar('pooled/average_error_' + str(i), avg_error_pool_0_t.mean())

                elif i == 1 and _ == 0:
                    sw_t.summ_scalar('unpooled/model_loss_4', model_loss)
                    model_loss_pool_4_t.update([model_loss.detach().cpu().numpy()])
                    sw_t.summ_scalar('pooled/model_loss_4', model_loss_pool_4_t.mean())

                    sw_t.summ_scalar('unpooled/mutual_loss_4', mutual_loss)
                    mutual_loss_pool_4_t.update([mutual_loss.detach().cpu().numpy()])
                    sw_t.summ_scalar('pooled/mutual_loss_4', mutual_loss_pool_4_t.mean())

                    sw_t.summ_scalar('unpooled/total_loss_4', total_loss)
                    loss_pool_4_t.update([total_loss.detach().cpu().numpy()])
                    sw_t.summ_scalar('pooled/total_loss_4', loss_pool_4_t.mean())

                    sw_t.summ_scalar('unpooled/average_error_4', ate)
                    avg_error_pool_4_t.update([ate.detach().cpu().numpy()])
                    sw_t.summ_scalar('pooled/average_error_4', avg_error_pool_4_t.mean())

                elif i == 2 and _ == 0:
                    sw_t.summ_scalar('unpooled/model_loss_8', model_loss)
                    model_loss_pool_8_t.update([model_loss.detach().cpu().numpy()])
                    sw_t.summ_scalar('pooled/model_loss_8', model_loss_pool_8_t.mean())

                    sw_t.summ_scalar('unpooled/mutual_loss_8', mutual_loss)
                    mutual_loss_pool_8_t.update([mutual_loss.detach().cpu().numpy()])
                    sw_t.summ_scalar('pooled/mutual_loss_8', mutual_loss_pool_8_t.mean())

                    sw_t.summ_scalar('unpooled/total_loss_8', total_loss)
                    loss_pool_8_t.update([total_loss.detach().cpu().numpy()])
                    sw_t.summ_scalar('pooled/total_loss_8', loss_pool_8_t.mean())

                    sw_t.summ_scalar('unpooled/average_error_8', ate)
                    avg_error_pool_8_t.update([ate.detach().cpu().numpy()])
                    sw_t.summ_scalar('pooled/average_error_8', avg_error_pool_8_t.mean())

            curr_e_traj = pred_traj

        total_loss = model_loss + mutual_loss

        total_loss.backward()
        optimizer.step()

        delta_avg_1 = compute_delta_avg(pred_traj, trajs, delta=1)
        delta_avg_4 = compute_delta_avg(pred_traj, trajs, delta=4)
        delta_avg_16 = compute_delta_avg(pred_traj, trajs, delta=16)

        sw_t.summ_scalar('delta_avg_1', delta_avg_1)
        delta_avg_1_pool_t.update([delta_avg_1.detach().cpu().numpy()])
        sw_t.summ_scalar('pooled/delta_avg_1', delta_avg_1_pool_t.mean())

        sw_t.summ_scalar('delta_avg_4', delta_avg_4)
        delta_avg_4_pool_t.update([delta_avg_4.detach().cpu().numpy()])
        sw_t.summ_scalar('pooled/delta_avg_4', delta_avg_4_pool_t.mean())

        sw_t.summ_scalar('delta_avg_16', delta_avg_16)
        delta_avg_16_pool_t.update([delta_avg_16.detach().cpu().numpy()])
        sw_t.summ_scalar('pooled/delta_avg_16', delta_avg_16_pool_t.mean())

        if sw_t is not None and sw_t.save_this:
            gt_rgb = utils.improc.preprocess_color(
                sw_t.summ_traj2ds_on_rgb('', target_traj[0:1], torch.mean(utils.improc.preprocess_color(rgbs[0:1]), dim=1),
                                       valids=torch.ones_like(valids[0:1]), cmap='winter', only_return=True))
            gt_black = utils.improc.preprocess_color(
                sw_t.summ_traj2ds_on_rgb('', target_traj[0:1], torch.ones_like(rgbs[0:1, 0]) * -0.5,
                                       valids=torch.ones_like(valids[0:1]),
                                       cmap='winter', only_return=True))
            sw_t.summ_traj2ds_on_rgb('outputs/single_trajs_on_gt_rgb', pred_traj[0:1], gt_rgb[0:1], cmap='spring')
            sw_t.summ_traj2ds_on_rgb('outputs/single_trajs_on_gt_black', pred_traj[0:1], gt_black[0:1], cmap='spring')

            sw_t.summ_rgbs('inputs_0/rgbs', utils.improc.preprocess_color(rgbs[0:1]).unbind(1))


        if do_val and (global_step) % val_freq == 0:
            torch.cuda.empty_cache()
            # let's do a val iter
            model.eval()
            sw_v = utils.improc.Summ_writer(
                writer=writer_v,
                global_step=global_step,
                log_freq=log_freq,
                fps=5,
                scalar_freq=1,
                just_gif=True)
            try:
                sample = next(val_iterloader)
            except StopIteration:
                val_iterloader = iter(val_dataloader)
                sample = next(val_iterloader)

            sample['rgbs'] = sample['rgbs'].cpu().detach().numpy()
            rgbs = torch.from_numpy(sample['rgbs']).cuda().float()  # B, S, C, H, W
            valids = sample['valids'].cuda().float()  # B, S, N

            with torch.no_grad():

                N0 = trajs.shape[2]
                target_idx = int(torch.randint(0, N0 - 1, (1,)))
                target_traj = trajs[:, :, target_idx:target_idx + 1, :]  # B, S, 1, 2

                target_e, trajs_e, vis_e = run_pips(pips, rgbs, target_traj, N, sw_t)  # N-2
                # mutual_loss = select_supporters(model, encoder, rgbs, trajs_e, target_e, vis_e)

                total_loss, ate, pred_tram = run_model(model, encoder, rgbs, trajs_e, target_traj,
                                            vis_e, criterion, sw_v)

                # total_loss = total_loss + mutual_loss

            sw_v.summ_scalar('total_loss', total_loss)
            loss_pool_v.update([total_loss.detach().cpu().numpy()])
            sw_v.summ_scalar('pooled/total_loss', loss_pool_v.mean())

            sw_v.summ_scalar('average_error', ate)
            avg_error_pool_v.update([ate.detach().cpu().numpy()])
            sw_v.summ_scalar('pooled/average_error', avg_error_pool_v.mean())

        iter_time = time.time() - iter_start_time
        print('%s; step %06d/%d; rtime %.2f; itime %.2f; loss = %.5f' % (
            model_name, global_step, max_iters, read_time, iter_time,
            total_loss.item()))

        if not global_step % save_freq:
            saverloader.save(ckpt_dir, optimizer, encoder, global_step, model_name='encoder')
            saverloader.save(ckpt_dir, optimizer, model, global_step, model_name='model')

    writer_t.close()
    if do_val:
        writer_v.close()


if __name__ == '__main__':
    # init argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_iters', type=int, help='iteration numbers',
                        default=20000)
    parser.add_argument('--use_cache', type=bool, help='whether to use cache in training;',
                        default=True)
    parser.add_argument('--cache_len', type=int, help='cache len',
                        default=10)
    args = parser.parse_args()

    cache = args.use_cache
    max_iters = args.max_iters
    cache_len = args.cache_len

    train()
