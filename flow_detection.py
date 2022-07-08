import numpy as np
from collections import Counter
import torch
import torch.nn as nn


def _nearest_pt(end_points, start_pt):
    pdist = nn.PairwiseDistance(p=2)
    start_pt = torch.reshape(start_pt, (1, -1))

    dist = pdist(end_points, start_pt)
    return end_points[int(torch.argmin(dist))]


def _split_trajs(S, sorted_list):
    # [256, 512, ...] demonstrate split points

    result_list = []
    end_ptr = 0
    trajs_total_num = len(sorted_list)

    for i in range(S):

        while 1:
            if end_ptr < trajs_total_num and sorted_list[end_ptr][-1] == i:
                end_ptr = end_ptr+1
            else:
                break

        result_list.append(end_ptr)

    return result_list


def flow_detect(sample, direction=True):
    # detect all flow in adjacent frames
    # return [[pt0, pt1], [, ]] where pt is tensor[2]

    if not direction:
        sample['trajs'] = torch.flip(sample['trajs'], [1])
    S, N = sample['trajs'].shape[1:3]
    flow_list = []

    for i in range(S-1):
        for j in range(N):
            # for pt_j in frame_i, find its best match point in frame i+1 as trajs' end point
            start_pt = sample['trajs'][0, i, j]
            end_pt = _nearest_pt(sample['trajs'][0, i+1], start_pt)
            flow_list.append([start_pt, end_pt, i])

    return flow_list


def linker(forward_flow_list, S, N, lifespan=3, consistency_threshold=0.5):
    # flow_list: every line is [pt0, pt1, begin_frame]
    # return: [N, lifespan+1], every line is a possible trajectory [pt0, pt1, ..., begin_frame]

    pdist = nn.PairwiseDistance(p=2)
    resutl_list = []
    forward_flow_list.sort(key=lambda x: x[-1])
    trajs_slice = _split_trajs(S, forward_flow_list) # [128, 256, 384, ...] for N=128


    # forward link process
    for begin_num in range(S-lifespan+1):
        # trajs_forward = [item for item in forward_flow_list if item[-1] == begin_num]

        start_slice = 0 if begin_num==0 else trajs_slice[begin_num-1]
        end_slice = trajs_slice[begin_num]

        trajs_forward = forward_flow_list[start_slice: end_slice]

        for begin_traj_idx in range(len(trajs_forward)):
            begin_traj = trajs_forward[begin_traj_idx]

            for shift in range(1, lifespan):
                possible_middle_point = forward_flow_list[trajs_slice[begin_num+shift-1]:
                                        trajs_slice[begin_num+shift]]

                possible_middle_point = [[*item, float(pdist(item[0], begin_traj[-2]))]
                                         for item in possible_middle_point
                                         if pdist(item[0], begin_traj[-2]) < consistency_threshold]

                if len(possible_middle_point):
                    possible_middle_point.sort(key=lambda x: x[-1])
                    extend_point = possible_middle_point[0][1]
                    begin_traj.insert(-2, extend_point)
                    trajs_forward[begin_traj_idx] = begin_traj

                else:
                    # trajs_forward.remove(begin_traj)
                    break

        resutl_list.extend([item for item in trajs_forward if len(item) == lifespan+1])

    return resutl_list

def trajs_compare(forward_link, backward_link, S):
    # return available trajectories

    # backward_reverse_link = [[ *reversed(trajs[0:lifespan]), S-trajs[-1] ] for trajs in backward_link]

    return forward_link