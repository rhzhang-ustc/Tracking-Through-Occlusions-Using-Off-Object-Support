import time
import argparse
import numpy as np
import timeit
import matplotlib
# import tensorflow as tf
# import scipy.misc
import io
import os
import math
from PIL import Image

matplotlib.use('Agg')  # suppress plot showing

import matplotlib.pyplot as plt

import matplotlib.animation as animation
import cv2

import utils.py
# import utils.box
import utils.misc
import utils.improc
# import utils.vox
import utils.grouping
from tqdm import tqdm
import random
import glob
# import color2d

from utils.basic import print_, print_stats

# import datasets
import flyingthingsdataset
# import cater_pointtraj_dataset

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter

import torch.nn.functional as F

# import inputs

device = 'cuda'
patch_size = 8
random.seed(125)
np.random.seed(125)


def balanced_ce_loss(pred, gt):
    # pred is B x 1 x Y x X
    pos = (gt > 0.95).float()
    neg = (gt < 0.05).float()

    label = pos * 2.0 - 1.0
    a = -label * pred
    b = F.relu(a)
    loss = b + torch.log(torch.exp(-b) + torch.exp(a - b))

    pos_loss = utils.basic.reduce_masked_mean(loss, pos)
    neg_loss = utils.basic.reduce_masked_mean(loss, neg)

    balanced_loss = pos_loss + neg_loss

    return balanced_loss, loss


def sequence_loss(flow_preds, flow_gt, vis, valids, gamma=0.8):
    """ Loss function defined over sequence of flow predictions """
    B, S, N, D = flow_gt.shape
    assert (D == 2)
    B, S, N = vis.shape
    B, S, N = valids.shape
    # print('flow_preds[0]', flow_preds[0].shape)
    # print('flow_gt', flow_gt.shape)
    # print('valid', valid.shape)

    n_predictions = len(flow_preds)
    flow_loss = 0.0
    # print('n_predictions', n_predictions)

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        fp = flow_preds[i]  # [:,:,0:1]
        T = fp.shape[1]
        i_loss = (fp - flow_gt[:, :T]).abs()  # B, S, N, 2
        i_loss = torch.mean(i_loss, dim=3)  # B, S, N
        # print('i_loss', i_loss.shape)
        # print('valid', valid.shape)
        # flow_loss += i_weight * i_loss.mean()
        # flow_loss += i_weight * (valid[:,:,None,None] * i_loss).mean()
        # flow_loss += i_weight * (valid * i_loss).mean()
        # flow_loss += i_weight * (valid * i_loss).mean()
        flow_loss += i_weight * utils.basic.reduce_masked_mean(i_loss, valids)

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=-1).sqrt()  # B, S, N
    # epe = epe.view(-1)
    # print('epe', epe.shape)
    # print('vis', vis.shape)

    # epe_vis = utils.basic.reduce_masked_mean(epe, valid.unsqueeze(1)*vis[:,1:])
    # epe_inv = utils.basic.reduce_masked_mean(epe, valid.unsqueeze(1)*(1.0-vis[:,1:]))
    epe_all = utils.basic.reduce_masked_mean(epe, valids)
    epe_vis = utils.basic.reduce_masked_mean(epe, valids * vis)
    epe_inv = utils.basic.reduce_masked_mean(epe, valids * (1.0 - vis))

    # epe_inv2inv = utils.basic.reduce_masked_mean(epe, valid.unsqueeze(1) * (1.0 - (vis[:,1:] + vis[:,:-1]).clamp(0,1)))
    epe_inv2inv = epe.mean() * 0

    metrics = {
        'epe': epe_all.mean().item(),
        'epe_vis': epe_vis.item(),
        'epe_inv': epe_inv.item(),
        'epe_inv2inv': epe_inv2inv.item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
        '10px': (epe < 10).float().mean().item(),
        '30px': (epe < 30).float().mean().item(),
    }

    return flow_loss, metrics


def run_model(d, N2, sw):
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)
    metrics = {
        'epe': 0,
        'epe_vis': 0,
        'epe_inv': 0,
        'epe_inv2inv': 0,
    }

    # flow = d['flow'].cuda().permute(0, 3, 1, 2)
    rgbs = torch.from_numpy(d['rgbs']).cuda().float()  # B, S, C, H, W
    masks = torch.from_numpy(d['masks']).cuda().float()  # B, S, C, H, W
    trajs = d['trajs'].cuda().float()  # B, S, N, 2
    vis = d['visibles'].cuda().float()  # B, S, N
    valids = d['valids'].cuda().float()  # B, S, N
    # updated_fails = d['updated_fails']
    # print('updated_fails', updated_fails)
    # sw.summ_scalar('updated_fails', updated_fails)

    B, S, C, H, W = rgbs.shape
    assert (C == 3)

    # print('rgbs', rgbs.shape)
    # print('trajs', trajs.shape)

    if sw is not None and sw.save_this:
        sw.summ_rgbs('inputs_0/rgbs', utils.improc.preprocess_color(rgbs[0:1]).unbind(1))
        sw.summ_traj2ds_on_rgbs('inputs_0/trajs_on_rgbs', trajs[0:1], utils.improc.preprocess_color(rgbs[0:1]),
                                valids=valids[0:1], cmap='winter')

    return total_loss, metrics

    # assert(B==1)
    if True:
        rgbs_flip = torch.flip(rgbs, [4])
        masks_flip = torch.flip(masks, [4])
        trajs_flip = trajs.clone()
        trajs_flip[:, :, :, 0] = W - 1 - trajs_flip[:, :, :, 0]
        vis_flip = vis.clone()
        valids_flip = valids.clone()
        trajs = torch.cat([trajs, trajs_flip], dim=0)
        vis = torch.cat([vis, vis_flip], dim=0)
        valids = torch.cat([valids, valids_flip], dim=0)
        rgbs = torch.cat([rgbs, rgbs_flip], dim=0)
        masks = torch.cat([masks, masks_flip], dim=0)
        B = B * 2

    if True:
        rgbs_flip = torch.flip(rgbs, [3])
        masks_flip = torch.flip(masks, [3])
        trajs_flip = trajs.clone()
        trajs_flip[:, :, :, 1] = H - 1 - trajs_flip[:, :, :, 1]
        vis_flip = vis.clone()
        valids_flip = valids.clone()
        trajs = torch.cat([trajs, trajs_flip], dim=0)
        vis = torch.cat([vis, vis_flip], dim=0)
        valids = torch.cat([valids, valids_flip], dim=0)
        rgbs = torch.cat([rgbs, rgbs_flip], dim=0)
        masks = torch.cat([masks, masks_flip], dim=0)
        B = B * 2

    # if True:
    #     B = 6
    #     rgbs = rgbs[:B]
    #     masks = masks[:B]
    #     vis = vis[:B]
    #     valids = valids[:B]
    #     trajs = trajs[:B]

    # if True:
    #     rgbs_ = rgbs.reshape(B*S, 3, H, W)
    #     masks_ = masks.reshape(B*S, 1, H, W)
    #     H_, W_ = 320, 512
    #     sy = H_/H
    #     sx = W_/W
    #     rgbs_ = F.interpolate(rgbs_, (320, 512), mode='bilinear')
    #     masks_ = F.interpolate(masks_, (320, 512), mode='bilinear')
    #     H, W = H_, W_
    #     rgbs = rgbs_.reshape(B, S, 3, H, W)
    #     masks = masks_.reshape(B, S, 1, H, W)
    #     trajs[:,:,:,0] *= sx
    #     trajs[:,:,:,1] *= sy

    # if sw is not None and sw.save_this:
    #     for b in range(B):
    #         sw.summ_rgb('inputs_0/rgb_b%d' % b, utils.improc.preprocess_color(rgbs[b:b+1,0]))
    #         sw.summ_oned('inputs_0/mask_b%d' % b, masks[b:b+1,0])
    #     for s in range(S):
    #         sw.summ_rgb('inputs_0/rgb_s%d' % s, utils.improc.preprocess_color(rgbs[0:1,s]))
    #         sw.summ_oned('inputs_0/mask_s%d' % s, masks[0:1,s])

    B, S, N, D = trajs.shape

    target_trajs = torch.zeros((B, S, N2, 2), dtype=torch.float32, device='cuda')
    target_vis = torch.zeros((B, S, N2), dtype=torch.float32, device='cuda')
    target_valids = torch.zeros((B, S, N2), dtype=torch.float32, device='cuda')

    for b in range(B):
        inds = np.random.choice(N, N2, replace=False)
        target_trajs[b] = trajs[b, :, inds]
        target_vis[b] = vis[b, :, inds]
        target_valids[b] = valids[b, :, inds]
    target_trajs = target_trajs.permute(0, 2, 1, 3).reshape(B * N2, S, 1, 2)
    target_vis = target_vis.permute(0, 2, 1).reshape(B * N2, S, 1)
    target_valids = target_valids.permute(0, 2, 1).reshape(B * N2, S, 1)
    # rgbs = rgbs.unsqueeze(1).repeat(1, N2, 1, 1, 1, 1).reshape(B*N2, S, 3, H, W)
    masks = masks.unsqueeze(1).repeat(1, N2, 1, 1, 1, 1).reshape(B * N2, S, 1, H, W)
    # N = N2
    B_bak = B
    N2_bak = N2

    B = B * N2
    N2 = 1

    _, S, _, H, W = rgbs.shape

    xy0 = target_trajs[:, 0, 0].round().long()  # B, 2
    singles = torch.ones_like(masks)
    for b in range(B):
        x, y = xy0[b, 0], xy0[b, 1]
        # print('y, x, H, W', y, x, H, W)
        assert (x >= 0 and x <= W - 1)
        assert (y >= 0 and y <= H - 1)
        id_ = masks[b, 0, 0, y, x]
        singles[b] = (masks[b] == id_).float()
    # print_stats('singles', singles)

    if sw is not None and sw.save_this:
        sw.summ_rgbs('inputs_0/rgbs', utils.improc.preprocess_color(rgbs[0:1]).unbind(1))
        sw.summ_oneds('inputs_0/masks', masks[0:1].unbind(1))

        # for b in range(3):
        #     sw.summ_traj2ds_on_rgb('inputs_0/target_%d_on_rgb' % b, target_trajs[b:b+1], torch.mean(utils.improc.preprocess_color(rgbs[b:b+1]), dim=1), cmap='winter')
        #     # sw.summ_oned('inputs_0/target_%d_masks' % b, torch.mean(singles[b:b+1], dim=1))

    # return total_loss, metrics

    # print('target_trajs', target_trajs.shape)
    # print('rgbs', rgbs.shape)
    preds, preds2, fcps, ccps, vis_e = model(target_trajs[:, 0], rgbs, coords_init=None, iters=6,
                                             sw=sw)  # list of B, N2, 2
    main_preds = [pred[:, :, 0:1] for pred in preds]
    loss, metrics = sequence_loss(main_preds, target_trajs, target_vis, target_valids, 0.8)
    total_loss += loss

    stride = 8
    H8, W8 = H // stride, W // stride

    cp_e = []
    argm = []
    for b in range(B):
        for s in range(S):
            cp = fcps[b, s, -1, 0]  # H8, W8
            xy = (target_trajs[b, s, 0] / stride).round().long()  # 2
            x, y = xy[0], xy[1]
            if (x >= 0 and
                    x <= W8 - 1 and
                    y >= 0 and
                    y <= H8 - 1 and
                    target_vis[b, s, 0] > 0 and
                    target_valids[b, s, 0] > 0
            ):
                heatmap_g = torch.zeros_like(cp)
                heatmap_g[y, x] = 1
                cp_e.append(cp.reshape(1, -1))
                cp_g = heatmap_g.reshape(1, -1)
                argm.append(torch.argmax(cp_g, dim=1))
    if len(cp_e):
        cp_e = torch.cat(cp_e, dim=0)
        argm = torch.cat(argm, dim=0)
        ce_loss = F.cross_entropy(cp_e, argm, reduction='mean')
        total_loss += ce_loss * 0
    else:
        ce_loss = total_loss * 0
    metrics['ce'] = ce_loss.item()

    if False:
        # print('ccps', ccps.shape)
        # print('singles', singles.shape)
        singles_ = F.interpolate(singles.reshape(B * S, 1, H, W), (H8, W8)).reshape(B, S, 1, H8, W8)
        coarse_loss, _ = balanced_ce_loss(ccps[:, :, -1], singles_)
        total_loss += coarse_loss
        metrics['co'] = coarse_loss.item()
    else:
        metrics['co'] = total_loss.item() * 0

    # print('vis_e', vis_e.shape)
    # print('vis', vis.shape)
    vis_loss = F.cross_entropy(vis_e.reshape(-1, 2), target_vis.reshape(-1).long(), reduction='mean')
    total_loss += vis_loss
    metrics['vis'] = vis_loss.item()

    trajs_e = preds[-1]
    trajs_e = trajs_e.reshape(B_bak, N2_bak, S, 2).permute(0, 2, 1, 3)
    target_trajs = target_trajs.reshape(B_bak, N2_bak, S, 2).permute(0, 2, 1, 3)
    target_valids = target_valids.reshape(B_bak, N2_bak, S).permute(0, 2, 1)
    # rgbs = rgbs.reshape(B_bak, N2_bak, S, 3, H, W)[:,0] # B, S, C, H, W

    if sw is not None and sw.save_this:

        sw.summ_traj2ds_on_rgbs('inputs_0/trajs_on_rgbs', target_trajs[0:1], utils.improc.preprocess_color(rgbs[0:1]),
                                valids=target_valids[0:1], cmap='winter')

        sw.summ_traj2ds_on_rgbs('outputs/trajs_on_rgbs', trajs_e[0:1], utils.improc.preprocess_color(rgbs[0:1]),
                                cmap='spring')
        sw.summ_traj2ds_on_rgbs('outputs/trajs_on_black', trajs_e[0:1], torch.ones_like(rgbs[0:1]) * -0.5,
                                cmap='spring')

        gt_rgb = utils.improc.preprocess_color(
            sw.summ_traj2ds_on_rgb('', target_trajs[0:1], torch.mean(utils.improc.preprocess_color(rgbs[0:1]), dim=1),
                                   valids=target_valids[0:1], cmap='winter', frame_id=metrics['epe'], only_return=True))
        gt_black = utils.improc.preprocess_color(
            sw.summ_traj2ds_on_rgb('', target_trajs[0:1], torch.ones_like(rgbs[0:1, 0]) * -0.5,
                                   valids=target_valids[0:1], cmap='winter', frame_id=metrics['epe'], only_return=True))
        sw.summ_traj2ds_on_rgb('outputs/single_trajs_on_gt_rgb', trajs_e[0:1], gt_rgb[0:1], cmap='spring')
        sw.summ_traj2ds_on_rgb('outputs/single_trajs_on_gt_black', trajs_e[0:1], gt_black[0:1], cmap='spring')

        rgb_vis = []
        black_vis = []
        for trajs_e in preds2:
            trajs_e = trajs_e[:B_bak * N2_bak]
            trajs_e = trajs_e.reshape(B_bak, N2_bak, S, 2).permute(0, 2, 1, 3)
            rgb_vis.append(sw.summ_traj2ds_on_rgb('', trajs_e[0:1], gt_rgb, only_return=True, cmap='spring'))
            black_vis.append(sw.summ_traj2ds_on_rgb('', trajs_e[0:1], gt_black, only_return=True, cmap='spring'))
        sw.summ_rgbs('outputs/animated_trajs_on_black', black_vis)
        sw.summ_rgbs('outputs/animated_trajs_on_rgb', rgb_vis)

    return total_loss, metrics


def train():
    # default coeffs (don't touch)
    init_dir = ''
    coeff_prob = 0.0
    use_augs = False

    # the idea here is to load and visualize the new clean data i exported
    exp_name = 'vd00'  # copy from single_point_track.py
    exp_name = 'vd01'  # quick; augs=False; show me
    exp_name = 'vd02'  # start_inds 0,1,2 (not 3)
    exp_name = 'vd03'  # fps
    exp_name = 'vd04'  # min 1000 bytes
    exp_name = 'vd05'  # check N earlier
    exp_name = 'vd06'  # req N > self.N*2 < yes this is faster than vd04
    exp_name = 'vd07'  # check load_fail at start of getitem
    exp_name = 'vd08'  # reduce code dup
    exp_name = 'vd09'  # return and plot updated_fails
    exp_name = 'vd10'  # clean up a bit
    exp_name = 'vd11'  # ab trajs
    exp_name = 'vd12'  # load some occ
    exp_name = 'vd13'  # occ version ac

    init_dir = ''

    ## choose hyps
    B = 1
    S = 8
    N = 256  # we need to load at least 4 i think
    N2 = 32
    lr = 1e-4
    grad_acc = 1

    # crop_size = (512,768)
    crop_size = (368, 496)
    # crop_size = (320,384)

    quick = True
    # quick = False
    if quick:
        max_iters = 10
        log_freq = 2
        save_freq = 99999999
        shuffle = True
        do_val = False
        cache = False
        cache_len = 101
        cache_freq = 99999999
        subset = 'A'
        use_augs = True
    else:
        max_iters = 100000
        log_freq = 500
        val_freq = 500
        save_freq = 1000
        shuffle = True
        do_val = False

        cache = False
        cache_len = 501
        cache_freq = 50000

        subset = 'all'
        use_augs = True

    # actual coeffs
    coeff_prob = 1.0

    ## autogen a name
    model_name = "%02d_%d_%d_%d" % (B, S, N, N2)
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

    ckpt_dir = 'checkpoints/%s' % model_name
    log_dir = 'logs_vis_data'
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)
    if do_val:
        writer_v = SummaryWriter(log_dir + '/' + model_name + '/v', max_queue=10, flush_secs=60)

    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    force_double_inb = False
    force_all_inb = False
    # train_dataset = flyingthingsdataset.FlyingThingsZigzagDataset(root_dir=root_dir, dset='TRAIN', subset=subset, use_augs=use_augs, N=N, S=8, S_out=S, crop_size=crop_size, version='ak', force_double_inb=force_double_inb, force_all_inb=force_all_inb)
    train_dataset = flyingthingsdataset.FlyingThingsDataset(
        dset='TRAIN', subset='A',
        use_augs=False, # use_augs,
        N=N, S=S,
        crop_size=crop_size,
        version='ab',
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
        # val_dataset = flyingthingsdataset.FlyingThingsZigzagDataset(root_dir=root_dir, dset='TEST', subset='all', use_augs=False, N=N, S=8, S_out=S, crop_size=crop_size, version='ak', force_double_inb=force_double_inb, force_all_inb=force_all_inb)
        val_dataset = flyingthingsdataset.FlyingThingsDataset(root_dir=root_dir, dset='TEST', subset='all',
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
    ce_pool_t = utils.misc.SimplePool(n_pool, version='np')
    co_pool_t = utils.misc.SimplePool(n_pool, version='np')
    vis_pool_t = utils.misc.SimplePool(n_pool, version='np')
    epe_pool_t = utils.misc.SimplePool(n_pool, version='np')
    epe_vis_pool_t = utils.misc.SimplePool(n_pool, version='np')
    epe_inv_pool_t = utils.misc.SimplePool(n_pool, version='np')
    epe_inv2inv_pool_t = utils.misc.SimplePool(n_pool, version='np')
    flow_pool_t = utils.misc.SimplePool(n_pool, version='np')
    if do_val:
        loss_pool_v = utils.misc.SimplePool(n_pool, version='np')
        ce_pool_v = utils.misc.SimplePool(n_pool, version='np')
        co_pool_v = utils.misc.SimplePool(n_pool, version='np')
        vis_pool_v = utils.misc.SimplePool(n_pool, version='np')
        epe_pool_v = utils.misc.SimplePool(n_pool, version='np')
        epe_vis_pool_v = utils.misc.SimplePool(n_pool, version='np')
        epe_inv_pool_v = utils.misc.SimplePool(n_pool, version='np')
        epe_inv2inv_pool_v = utils.misc.SimplePool(n_pool, version='np')
        flow_pool_v = utils.misc.SimplePool(n_pool, version='np')

    while global_step < max_iters:

        read_start_time = time.time()

        global_step += 1
        total_loss = torch.tensor(0.0, requires_grad=True).to(device)

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

        total_loss, metrics = run_model(sample, N2, sw_t)

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
                total_loss, metrics = run_model(sample, N2, sw_v)
            sw_v.summ_scalar('total_loss', total_loss)
            loss_pool_v.update([total_loss.detach().cpu().numpy()])
            sw_v.summ_scalar('pooled/total_loss', loss_pool_v.mean())

        iter_time = time.time() - iter_start_time
        print('%s; step %06d/%d; rtime %.2f; itime %.2f; loss = %.5f' % (
            model_name, global_step, max_iters, read_time, iter_time,
            total_loss.item()))

    writer_t.close()
    if do_val:
        writer_v.close()


if __name__ == '__main__':
    train()
