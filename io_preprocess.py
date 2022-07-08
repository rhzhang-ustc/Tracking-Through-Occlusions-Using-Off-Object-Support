import numpy as np
import torch
import torch.nn as nn


def build_input_matrix(sample, trajs):
  # sample:
  # trajs: [[], []] every line has shape[pt0, pt1, ...., begin_frams]

  trajs.sort(key=lambda x: x[-1])

  S, B, C, H, W = sample['rgbs'].size()
  frames_concat = sample['rgbs'].reshape(-1, H, W)
  N = len(trajs)  # (N, C) where N is the number of features

  for i in range(N):
    traj = trajs[i]
    pt_x, pt_y, pt_t = float(traj[0][0]), float(traj[0][1]), traj[-1]

