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
N = 256 + 1  # we need to load at least 4 i think
lr = 1e-5
grad_acc = 1

crop_size = (368, 496)

max_iters = 10000
log_freq = 1000
save_freq = 5000
shuffle = True
do_val = True

cache = True
cache_len = 100
cache_freq = 99999999
use_augs = True

val_freq = 50

model_depth = 1

queries_dim = 27 + 2
dim = 3
feature_map_dim = 128
encoder_stride = 8

embedding_dim = 1024    # dimension of joint embedding space

num_band = 32
k = 10  # supervision
k_vis = 100  # visualize
feature_sample_step = 1
vis_threshold = 0.1

beta = 3 # importance score of top k loss
alpha = 3

init_dir = 'reference_model'

log_dir = 'test_logs'
model_name_suffix = 'simple_mlp'
ckpt_dir = 'checkpoints_simple'
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


def run_pips(model, rgbs, N, sw):
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

    preds, preds_anim, vis_e, stats = model(xy, rgbs, iters=6)
    trajs_e = preds[-1]

    pad = 50
    rgbs = F.pad(rgbs.reshape(B * S, 3, H, W), (pad, pad, pad, pad), 'constant', 0).reshape(B, S, 3, H + pad * 2,
                                                                                            W + pad * 2)
    trajs_e = trajs_e + pad

    return trajs_e - pad, vis_e


def run_model(embedding_mlp,
              voting_mlp,
              xy_mlp,
              GRU,
              encoder,
              rgbs, trajs, target_traj,
              valids, criterion, sw):

    total_loss = torch.tensor(0.0, requires_grad=True, device=device)

    B, S, C, H, W = rgbs.shape
    _, _, N0, _ = trajs.shape

    pred_traj = torch.zeros(B, S, 1, 2).cuda()

    frame_features_map = encoder(rgbs.reshape(B * S, C, H, W)).reshape(B, S, feature_map_dim,
                                                                       H // encoder_stride,
                                                                       W // encoder_stride)  # 1, 8, 128, 46, 62

    '''
    generate target feature
    '''
    start_target_pos = target_traj[:, 0, :, :]  # B, 1, 2
    target_pos_encoding = utils.misc.get_2d_embedding(start_target_pos, num_band)  # B, 1, 66

    target_motion = (target_traj[:, 1, :, :] - start_target_pos)  # B, 1, 2
    target_motion_encoding = utils.misc.get_2d_embedding(target_motion, num_band)   # B, 1, 66

    target_frame_feature = utils.samp.bilinear_sample2d(frame_features_map[:, 0, :],
                                                        start_target_pos[:, 0, 0:1] / encoder_stride,
                                                        start_target_pos[:, 0, 1:2] / encoder_stride).permute(0, 2, 1)  # B, 1, 128

    target_feature = torch.concat([target_pos_encoding, target_motion_encoding,
                                   torch.ones(B, 1, 1).cuda(),
                                   target_frame_feature,
                                   ], dim=-1)
    target_embedding = embedding_mlp(target_feature)  # B, 1, 1024

    '''
    get supporters' feature
    '''

    h = target_embedding.clone().detach().requires_grad_(True)

    for s in range(S):

        pos = trajs[:, s, :, :]  # B, N0, 2
        pos_encoding = utils.misc.get_2d_embedding(pos.reshape(B, -1, 2), num_band)  # B, N0, 66

        if s != S - 1:
            motion = trajs[:, s + 1, :, :] - pos
        else:
            motion = torch.mean(trajs, dim=1) - pos
        motion_encoding = utils.misc.get_2d_embedding(motion.reshape(B, -1, 2), num_band)  # B, N0, 66

        features = utils.samp.bilinear_sample2d(frame_features_map[:, s, :],
                                                pos[:, :, 0] / encoder_stride,
                                                pos[:, :, 1] / encoder_stride).permute(0, 2, 1)  # B, N0, 128

        supporters_features = torch.concat([pos_encoding, motion_encoding, valids[:, s, :, None], features], dim=-1)

        supporters_embedding = embedding_mlp(supporters_features)

        pred = voting_mlp(torch.concat([supporters_embedding, target_embedding.repeat(1, N0, 1)], dim=-1))  # B, N0, embedding_dim+1

        w = pred[:, :, -1:]  # B, N0, 1  weight for vote that begins at start point
        norm_w = torch.softmax(w, dim=1)  # B, N0, 1

        votes_embedding = pred[:, :, :-1]
        aggregate_votes_embedding = torch.sum(votes_embedding * norm_w, dim=-2).reshape(B, 1, -1)   # B, 1, embedding_dim

        '''
        update target feature
        '''
        target_embedding, h = GRU(aggregate_votes_embedding, h)

        xy = xy_mlp(target_embedding)  # B, 1, 2
        supporters_xy = xy_mlp(votes_embedding)     # B, N0, 2

        pred_traj[:, s, :, :] = xy

        target_traj_s = target_traj[:, s, :, :]  # B, 1, 2

        total_loss = total_loss + criterion(xy, target_traj_s)

        '''
        additional loss: force topk points to have reasonable result
        '''
        try:
            k_temp = k_vis
            _, top_k_index = torch.topk(norm_w, k_temp, dim=1)
        except Exception:
            k_temp = len(norm_w)
            _, top_k_index = torch.topk(norm_w, k_temp, dim=1)  # B, 100, 1

        pred_matrix = supporters_xy.index_select(1, top_k_index[0, :, 0])

        # more easy way
        if k > k_temp:
            k_i = k_temp  # number of supporters that recieve loss
        else:
            k_i = k

        total_loss = total_loss + beta * criterion(pred_matrix[:, :k_i, :],
                                                   target_traj_s.repeat(1, k_i, 1))  # loss for k

    avg_traj_error = torch.mean(torch.norm(pred_traj-target_traj, 2))

    if sw is not None and sw.save_this:
        gt_rgb = utils.improc.preprocess_color(
            sw.summ_traj2ds_on_rgb('', target_traj[0:1], torch.mean(utils.improc.preprocess_color(rgbs[0:1]), dim=1),
                                   valids=torch.ones_like(valids[0:1]), cmap='winter', only_return=True))
        gt_black = utils.improc.preprocess_color(
            sw.summ_traj2ds_on_rgb('', target_traj[0:1], torch.ones_like(rgbs[0:1, 0]) * -0.5,
                                   valids=torch.ones_like(valids[0:1]),
                                   cmap='winter', only_return=True))
        sw.summ_traj2ds_on_rgb('outputs/single_trajs_on_gt_rgb', pred_traj[0:1], gt_rgb[0:1], cmap='spring')
        sw.summ_traj2ds_on_rgb('outputs/single_trajs_on_gt_black', pred_traj[0:1], gt_black[0:1], cmap='spring')

        sw.summ_rgbs('inputs_0/rgbs', utils.improc.preprocess_color(rgbs[0:1]).unbind(1))

        sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs',
                                trajs,
                                utils.improc.preprocess_color(rgbs[0:1]),
                                cmap='spring')

        sw.summ_feats('tff/0_fmaps', frame_features_map.unbind(1))

    return total_loss / S, \
           avg_traj_error, \
           pred_traj


def train():
    # model save path
    # model_path = 'checkpoints/01_8_64_32_1e-4_p1_avg_trajs_20:44:39.pth'  # where the ckpt is
    # state = torch.load(model_path)
    # model.load_state_dict(state['model_state'])

    # actual coeffs
    coeff_prob = 1.0

    ## autogen a name
    exp_name = 'simple'
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
            shuffle=False,
            num_workers=12,
            pin_memory=True)
        val_iterloader = iter(val_dataloader)

    global_step = 0

    n_pool = 100
    loss_pool_t = utils.misc.SimplePool(n_pool, version='np')
    avg_error_pool_t = utils.misc.SimplePool(n_pool, version='np')

    n_pool_v = 100
    if do_val:
        loss_pool_v = utils.misc.SimplePool(n_pool_v, version='np')
        avg_error_pool_v = utils.misc.SimplePool(n_pool, version='np')

    total_loss = torch.tensor(0.0, requires_grad=True).to(device)

    # we need four model: embedding_mlp, voting_mlp, xy_mlp and encoder(cnn)
    embedding_mlp = MLP(in_dim=2 * (2 * num_band + 2) + feature_map_dim + 1,
                        out_dim=embedding_dim,
                        hidden_dim=1024
    )

    embedding_mlp = embedding_mlp.to(device)

    voting_mlp = MLP(in_dim=2 * embedding_dim,
                     out_dim=embedding_dim + 1,
                     hidden_dim=1024
    )

    voting_mlp = voting_mlp.to(device)

    xy_mlp = torch.nn.Linear(embedding_dim, 2)
    xy_mlp = xy_mlp.to(device)

    GRU = torch.nn.GRU(input_size=embedding_dim, hidden_size=embedding_dim)
    GRU = GRU.to(device)

    encoder = BasicEncoder(input_dim=3, output_dim=feature_map_dim, stride=encoder_stride)
    encoder = encoder.to(device)

    pips = Pips(stride=4).cuda()
    saverloader.load(init_dir, pips)
    pips.eval()

    if use_ckpt:

        saverloader.load(ckpt_dir, embedding_mlp, optimizer=None, scheduler=None, model_ema=None, step=0,
                         model_name='embedding_mlp',
                         ignore_load=None)
        saverloader.load(ckpt_dir, voting_mlp, optimizer=None, scheduler=None, model_ema=None, step=0,
                         model_name='voting_mlp',
                         ignore_load=None)
        saverloader.load(ckpt_dir, encoder, optimizer=None, scheduler=None, model_ema=None, step=0,
                         model_name='encoder',
                         ignore_load=None)
        saverloader.load(ckpt_dir, xy_mlp, optimizer=None, scheduler=None, model_ema=None, step=0,
                         model_name='xy_mlp',
                         ignore_load=None)

    criterion = nn.L1Loss()
    optimizer = optim.AdamW([
        {'params': embedding_mlp.parameters(), 'lr': lr},
        {'params': voting_mlp.parameters(), 'lr': lr},
        {'params': encoder.parameters(), 'lr': lr},
        {'params': xy_mlp.parameters(), 'lr': lr},
    ])

    while global_step < max_iters:

        embedding_mlp.train()
        voting_mlp.train()
        encoder.train()

        read_start_time = time.time()
        global_step += 1

        sw_t = utils.improc.Summ_writer(
            writer=writer_t,
            global_step=global_step,
            log_freq=log_freq,
            fps=3,
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
        trajs = sample['trajs'].cuda().float()  # B, S, N, 2

        '''
        train process
        '''

        read_time = time.time() - read_start_time
        iter_start_time = time.time()

        optimizer.zero_grad()

        trajs_e, vis_e = run_pips(pips, rgbs, N, sw_t)

        N0 = trajs.shape[2]     # total number of trajectories
        target_idx = int(torch.randint(0, N0 - 1, (1,)))
        target_traj = trajs[:, :, target_idx:target_idx+1, :]  # B, S, 1, 2

        total_loss, avg_error, _ = run_model(embedding_mlp,
                                             voting_mlp,
                                             xy_mlp,
                                             GRU,
                                             encoder,
                                             rgbs, trajs_e, target_traj,
                                             vis_e, criterion, sw_t)

        total_loss.backward()
        optimizer.step()

        sw_t.summ_scalar('total_loss', total_loss)
        loss_pool_t.update([total_loss.detach().cpu().numpy()])
        sw_t.summ_scalar('pooled/total_loss', loss_pool_t.mean())

        sw_t.summ_scalar('average_error', avg_error)
        avg_error_pool_t.update([avg_error.detach().cpu().numpy()])
        sw_t.summ_scalar('pooled/average_error', avg_error_pool_t.mean())

        if do_val and (global_step) % val_freq == 0:
            torch.cuda.empty_cache()
            # let's do a val iter

            embedding_mlp.eval()
            voting_mlp.eval()
            encoder.eval()

            sw_v = utils.improc.Summ_writer(
                writer=writer_v,
                global_step=global_step,
                log_freq=log_freq,
                fps=3,
                scalar_freq=2,
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
                trajs_e, vis_e = run_pips(pips, rgbs, N, sw_t)

                target_traj = trajs_e[:, :, 0:1, :]  # B, S, 1, 2
                trajs_e = trajs_e[:, :, 1:, :]  # B, S, N0, 2
                vis_e = vis_e[:, :, 1:]

                total_loss,  avg_error, _ = run_model(embedding_mlp,
                                                      voting_mlp,
                                                      xy_mlp,
                                                      GRU,
                                                      encoder,
                                                      rgbs, trajs_e, target_traj,
                                                      vis_e, criterion, sw_v)

            sw_v.summ_scalar('total_loss', total_loss)
            loss_pool_v.update([total_loss.detach().cpu().numpy()])
            sw_v.summ_scalar('pooled/total_loss', loss_pool_v.mean())

            sw_v.summ_scalar('average_error', avg_error)
            avg_error_pool_v.update([avg_error.detach().cpu().numpy()])
            sw_v.summ_scalar('pooled/average_error', avg_error_pool_v.mean())

        iter_time = time.time() - iter_start_time
        print('%s; step %06d/%d; rtime %.2f; itime %.2f; loss = %.5f' % (
            model_name, global_step, max_iters, read_time, iter_time,
            total_loss.item()))

        if not global_step % save_freq:
            saverloader.save(ckpt_dir, optimizer, encoder, global_step, model_name='encoder')
            saverloader.save(ckpt_dir, optimizer, embedding_mlp, global_step, model_name='embedding_mlp')
            saverloader.save(ckpt_dir, optimizer, voting_mlp, global_step, model_name='voting_mlp')
            saverloader.save(ckpt_dir, optimizer, xy_mlp, global_step, model_name='xy_mlp')

    writer_t.close()
    if do_val:
        writer_v.close()


if __name__ == '__main__':
    # init argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_iters', type=int, help='iteration numbers',
                        default=20000)
    parser.add_argument('--use_cache', type=bool, help='whether to use cache in training;',
                        default=False)
    parser.add_argument('--cache_len', type=int, help='cache len',
                        default=100)
    args = parser.parse_args()

    cache = args.use_cache
    max_iters = args.max_iters
    cache_len = args.cache_len

    train()
