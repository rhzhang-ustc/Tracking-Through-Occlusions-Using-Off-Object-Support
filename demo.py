import time
import numpy as np
import matplotlib
import os
import glob
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

matplotlib.use('Agg')  # suppress plot showing
import utils.py
import utils.misc
import utils.improc
import utils.grouping
import utils.samp
import utils.basic
import random

import torch
import torch.nn as nn
import saverloader

from tensorboardX import SummaryWriter
import imageio.v2 as imageio

from net.perceiver_pytorch import MLP
from net.pips import BasicEncoder, Pips
from train_mlp_full_trajs import run_pips, run_model

device = 'cuda'
device_ids = [0]
patch_size = 8
random.seed(125)
np.random.seed(125)

## choose hyps
B = 1
S = 8
N = 256  # we need to load at least 4 i think

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

init_dir = 'reference_model'
ckpt_dir = 'checkpoints_v1_rand'

video_path = './demo_videos/ethCup_input.mp4'

num_worker = 12

log_dir = 'demo_logs'
# target_0 = (160, 168)     # center of the cup, H, W
target_0 = (265, 170)


def split_frames(video_path, output_path, start, end, step):

    cap = cv2.VideoCapture(video_path)
    count = 0

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, c = np.array(frame).shape
        if start <= count < end and (count - start) % step == 0:

            # cv2.circle(frame, (int(target_0[0]), int(target_0[1])), 3, (0, 0, 255), thickness=-1)
            frame = cv2.resize(frame, (496, 368))
            cv2.imwrite(output_path + '/' + '%04d' % count + '.jpg', frame)
        count += 1


def test_model():
    # the idea in this file is to run the model on some demo images, and return some visualizations

    exp_name = '00'  # (exp_name is used for logging notes that correspond to different runs)

    split_frames(video_path, './demo_images', 96, 1000, 6)

    filenames = glob.glob('./demo_images/*.jpg')
    filenames = sorted(filenames)
    print('filenames', filenames)
    max_iters = len(filenames) // S  # run each unique subsequence

    log_freq = 1  # when to produce visualizations

    ## autogen a name
    model_name = "%02d_%d_%d" % (B, S, N)
    model_name += "_%s" % exp_name
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)

    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)

    model = MLP(in_dim=S * (3 * num_band + 3) + 2 * feature_map_dim + (S-1) * (2 * num_band + 2),
                out_dim=S * 3,
                hidden_dim=4096
    )

    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)

    encoder = BasicEncoder(input_dim=3, output_dim=feature_map_dim, stride=encoder_stride)
    encoder = encoder.to(device)
    encoder = torch.nn.DataParallel(encoder, device_ids=device_ids)

    pips = Pips(stride=4).to(device)
    saverloader.load(init_dir, pips)
    pips.eval()

    saverloader.load(ckpt_dir, model, optimizer=None, scheduler=None, model_ema=None, step=0, model_name='model',
                     ignore_load=None)
    saverloader.load(ckpt_dir, encoder, optimizer=None, scheduler=None, model_ema=None, step=0, model_name='encoder',
                     ignore_load=None)

    valids = torch.ones(B, S, N).cuda().float()  # B, S, N
    criterion = nn.L1Loss()

    target = torch.Tensor(target_0).reshape(B, 1, 1, 2).repeat(1, S, 1, 1).float().to(device)   # B, 1, 1, 2

    global_step = 0

    while global_step < max_iters:

        read_start_time = time.time()

        global_step += 1

        sw_t = utils.improc.Summ_writer(
            writer=writer_t,
            global_step=global_step,
            log_freq=log_freq,
            fps=5,
            scalar_freq=int(log_freq / 2),
            just_gif=True)

        try:
            rgbs = []
            for s in range(S):
                fn = filenames[(global_step - 1) * S + s]
                if s == 0:
                    print('start frame', fn)
                im = imageio.imread(fn)
                im = im.astype(np.uint8)
                rgbs.append(torch.from_numpy(im).permute(2, 0, 1))

            rgbs = torch.stack(rgbs, dim=0).unsqueeze(0).to(device).float()  # 1, S, C, H, W
            rgbs = rgbs.flip(2)

            read_time = time.time() - read_start_time
            iter_start_time = time.time()

            with torch.no_grad():

                trajs_e, vis_e = run_pips(pips, rgbs, N, sw_t)
                trajs_e = trajs_e.to(device)    # B, S, N, 2
                vis_e = vis_e.to(device)

                if global_step == 1:
                    near_target = torch.argmin(torch.sum(torch.abs(trajs_e-target), dim=-1), dim=-1)   # B, S
                    near_idx = near_target[0, 0]
                    target = trajs_e[0, :, near_idx, :].reshape(B, S, 1, 2)
                    print(target)

                _, _, _, pred_traj = run_model(model, encoder, rgbs, trajs_e, target, valids, criterion, sw_t)
                target = pred_traj[:, -1, :, :].repeat(1, S, 1, 1).to(device)

            iter_time = time.time() - iter_start_time
            print('%s; step %06d/%d; rtime %.2f; itime %.2f' % (
                model_name, global_step, max_iters, read_time, iter_time))
        except FileNotFoundError as e:
            print('error', e)

    writer_t.close()

if __name__ == '__main__':
    test_model()