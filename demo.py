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
vis_threshold = 0.01

init_dir = 'reference_model'
num_worker = 12

log_dir = 'demo_logs'

model_path = 'checkpoints_test/01_8_257_1e-5_p1_traj_estimation_00:58:10_selection_model.pth'  # where the ckpt is
encoder_path = 'checkpoints_test/01_8_257_1e-5_p1_traj_estimation_00:58:10_selection_encoder.pth'

target_0 = (95, 109)     # center of the cup


def split_frames(video_path, output_path, start, end, step):

    cap = cv2.VideoCapture(video_path)
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, c = np.array(frame).shape
        frame = cv2.resize(frame, dsize=(int(w/2), int(h/2)))
        if start <= count < end and (count - start) % step == 0:
            cv2.imwrite(output_path + '/' + '%04d' % count + '.jpg', frame)
        count += 1


def test_model():
    # the idea in this file is to run the model on some demo images, and return some visualizations

    exp_name = '00'  # (exp_name is used for logging notes that correspond to different runs)

    split_frames('./demo_videos/ethCup_input.mp4', './demo_images', 0, 1000, 1)

    filenames = glob.glob('./demo_images/*.jpg')
    filenames = sorted(filenames)
    print('filenames', filenames)
    max_iters = len(filenames) // S  # run each unique subsequence

    log_freq = 2  # when to produce visualizations

    ## autogen a name
    model_name = "%02d_%d_%d" % (B, S, N)
    model_name += "_%s" % exp_name
    import datetime
    model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name', model_name)

    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)

    model = MLP(in_dim=2 * S * ((3 * num_band + 3) + feature_map_dim), out_dim=S * 3)
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)

    encoder = BasicEncoder(input_dim=3, output_dim=feature_map_dim, stride=encoder_stride)
    encoder = encoder.to(device)
    encoder = torch.nn.DataParallel(encoder, device_ids=device_ids)

    pips = Pips(stride=4).to(device)
    saverloader.load(init_dir, pips)
    pips.eval()

    model_state = torch.load(model_path)
    encoder_state = torch.load(encoder_path)

    model.load_state_dict(model_state, strict=False)
    encoder.load_state_dict(encoder_state, strict=False)

    valids = torch.ones(B, S, N).cuda().float()  # B, S, N
    criterion = nn.L1Loss()

    target = torch.Tensor(target_0).reshape(B, 1, 1, 2).repeat(1, S, 1, 1).float().to(device)

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

            read_time = time.time() - read_start_time
            iter_start_time = time.time()

            with torch.no_grad():
                trajs_e = run_pips(pips, rgbs, N, sw_t).to(device)
                _, _, pred_traj = run_model(model, encoder, rgbs, trajs_e, target, valids, criterion, sw_t)
                target = pred_traj[:, -1, :, :].repeat(1, S, 1, 1).to(device)

            iter_time = time.time() - iter_start_time
            print('%s; step %06d/%d; rtime %.2f; itime %.2f' % (
                model_name, global_step, max_iters, read_time, iter_time))
        except FileNotFoundError as e:
            print('error', e)

    writer_t.close()

if __name__ == '__main__':
    test_model()