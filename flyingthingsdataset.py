from numpy import random
from numpy.core.numeric import full
import torch
import numpy as np
import os
import scipy.ndimage
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import random

from torch._C import dtype, set_flush_denormal
# from detectron2.structures.masks import polygons_to_bitmask

import utils.py
import utils.basic
import utils.geom
import utils.improc

import glob
import json

import imageio
import cv2
import re
# import skimage.morphology
import sys

from torchvision.transforms import ColorJitter, GaussianBlur
import copy

np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')


def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data

def readImage(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        data = readPFM(name)
        if len(data.shape)==3:
            return data[:,:,0:3]
        else:
            return data
    return imageio.imread(name)


class FlyingThingsDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_location='../../aharley/flyingthings',
                 dset='TRAIN',
                 subset='all',
                 use_augs=False,
                 zigzag=False,
                 N=0, S=8,
                 S_out=8,
                 crop_size=(368, 496),
                 version='ad',
                 occ_version='al',
                 force_twice_vis=True,
                 N_min=None):

        print('loading FlyingThingsDataset...')

        
        self.S = S
        self.N = N
        if N_min is None:
            self.N_min = self.N//4
        else:
            self.N_min = N_min
            
        self.use_augs = use_augs
        self.zigzag = zigzag
        self.S_out = S_out

        self.rgb_paths = []
        self.traj_paths = []
        self.mask_paths = []
        self.flow_f_paths = []
        self.flow_b_paths = []
        self.start_inds = []
        self.load_fails = []

        # self.force_inb = force_inb
        # self.force_double_inb = force_double_inb
        self.force_twice_vis = force_twice_vis
        # self.force_all_inb = force_all_inb

        self.subset = subset

        if self.subset=='all':
            subsets = ['A', 'B', 'C']
        else:
            subsets = [subset]

        for subset in subsets:
            rgb_root_path = os.path.join(dataset_location, "frames_cleanpass_webp", dset, subset)
            flow_root_path = os.path.join(dataset_location, "optical_flow", dset, subset)
            traj_root_path = os.path.join(dataset_location, "trajs_%s" % version, dset, subset)
            mask_root_path = os.path.join(dataset_location, "object_index", dset, subset)
            # heavy_root_path = os.path.join(dataset_location, "heavy_raft_flows_ab", dset, subset)

            folder_names = [folder.split('/')[-1] for folder in glob.glob(os.path.join(traj_root_path, "*"))]
            folder_names = sorted(folder_names)
            # print('first 10 folders only')
            # folder_names = folder_names[:10]
            # print('folder_names', folder_names)

            for ii, folder_name in enumerate(folder_names):
                for lr in ['left', 'right']:
                    cur_rgb_path = os.path.join(rgb_root_path, folder_name, lr)
                    cur_traj_path = os.path.join(traj_root_path, folder_name, lr)
                    cur_mask_path = os.path.join(mask_root_path, folder_name, lr)

                    for start_ind in [0,1,2,3]:
                        traj_fn = cur_traj_path + '/trajs_at_%d.npz' % start_ind
                        if os.path.isfile(traj_fn):
                            file_size = os.path.getsize(traj_fn)
                            if file_size > 1000: # the empty ones are 264 bytes
                                # trajs = np.load(os.path.join(cur_traj_path, 'trajs_at_%d.npz' % start_ind), allow_pickle=True)
                                # trajs = dict(trajs)['trajs'] # S,N,2
                                # S, N, D = trajs.shape
                                # if N >= self.N:
                                #     # print('adding this one')
                                #     self.rgb_paths.append(cur_rgb_path)
                                #     self.traj_paths.append(cur_traj_path)
                                #     self.mask_paths.append(cur_mask_path)
                                #     self.start_inds.append(start_ind)
                                #     self.load_fails.append(0)
                                #     sys.stdout.write('.')
                                #     sys.stdout.flush()
                                # else:
                                #     sys.stdout.write('l')
                                #     sys.stdout.flush()

                                self.rgb_paths.append(cur_rgb_path)
                                self.traj_paths.append(cur_traj_path)
                                self.mask_paths.append(cur_mask_path)
                                self.start_inds.append(start_ind)
                                self.load_fails.append(0)
                                sys.stdout.write('.')
                                sys.stdout.flush()
                            
                # if ii > 10 and (ii % 100)==0:
                #     # print('found %d samples in %s so far...' % (len(self.rgb_paths), dataset_location))
                #     sys.stdout.write('%.1f%%' % (100*(ii+1)/len(folder_names)))
                #     sys.stdout.flush()
        # print('%.1f%%' % (100*(ii+1)/len(folder_names)))
        print('found %d samples in %s (dset=%s, subset=%s, version=%s)' % (len(self.rgb_paths), dataset_location, dset, self.subset, version))


        # we also need to step through and collect ooccluder info
        print('loading occluders...')

        self.occ_rgb_paths = []
        self.occ_mask_paths = []
        self.occ_start_inds = []
        self.occ_traj_paths = []

        # print('locking start_ind=0, for speed')
        for subset in subsets:
            # print('sub')
            # print(subset*10)

            rgb_root_path = os.path.join(dataset_location, "frames_cleanpass_webp", dset, subset)
            flow_root_path = os.path.join(dataset_location, "optical_flow", dset, subset)
            mask_root_path = os.path.join(dataset_location, "object_index", dset, subset)
            occ_root_path = os.path.join(dataset_location, "occluders_%s" % occ_version, dset, subset)
            
            # print('occ_root_path', occ_root_path)

            folder_names = [folder.split('/')[-1] for folder in glob.glob(os.path.join(occ_root_path, "*"))]
            folder_names = sorted(folder_names)
            # print('folder_names', folder_names)
            # print('first 10 folders only')
            # folder_names = folder_names[:10]

            # print('folder_names', folder_names)
            
            # for ii, folder_name in enumerate(folder_names):
            for folder_name in folder_names:
                
                for lr in ['left', 'right']:

                    cur_rgb_path = os.path.join(rgb_root_path, folder_name, lr)
                    cur_mask_path = os.path.join(mask_root_path, folder_name, lr)
                    cur_occ_path = os.path.join(occ_root_path, folder_name, lr)
                    
                    # start_ind = 0
                    # if True:
                    for start_ind in [0,1,2]:
                        occ_fn = cur_occ_path + '/occluder_at_%d.npy' % (start_ind)

                        # print("occ_fn', occ_fn')
                        # # file_names = glob.glob(os.path.join(cur_occ_path, "*at_%d.npz" % start_ind))
                        # file_names = glob.glob(os.path.join(cur_occ_path, "*at_%d.npy" % start_ind))
                        # print('file_names', file_names)
                        # print('len', len(file_names))
                        # # input()
                                               
                        # for ii in range(len(file_names)): 
                        if os.path.isfile(occ_fn):
                            file_size = os.path.getsize(occ_fn)
                            if file_size > 1000: # the empty ones are 10 bytes


                                # print('file_size', file_size)

                                # occ_info = np.load(occ_fn, allow_pickle=True)
                                # occ_info = dict(occ_info)
                                # occ_trajs = occ_info['trajs'] # S,N,2, or None
                                # occ_id = occ_info['id_'] # []
                                # print('occ_trajs', occ_trajs.shape)

                                self.occ_rgb_paths.append(cur_rgb_path)
                                self.occ_mask_paths.append(cur_mask_path)
                                self.occ_start_inds.append(start_ind)
                                self.occ_traj_paths.append(occ_fn)
                                # print('adding something')

                        sys.stdout.write('.')
                        sys.stdout.flush()
                            
                # if ii > 10 and (ii % 100)==0:
                #     # print('found %d samples in %s so far...' % (len(self.rgb_paths), dataset_location))
                #     sys.stdout.write('%.1f%%' % (100*(ii+1)/len(folder_names)))
                #     sys.stdout.flush()
        # print('%.1f%%' % (100*(ii+1)/len(folder_names)))
        print('found %d occluders in %s (dset=%s, subset=%s, version=%s)' % (len(self.occ_rgb_paths), dataset_location, dset, self.subset, occ_version))

        # print('self.occ_rgb_paths', self.occ_rgb_paths[:5])
        # print('self.occ_traj_paths', self.occ_traj_paths[:5])
        

        # photometric augmentation
        # self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.photo_aug = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.25/3.14)
        self.blur_aug = GaussianBlur(11, sigma=(0.1, 2.0))
        
        self.blur_aug_prob = 0.2
        self.color_aug_prob = 0.5

        # occlusion augmentation
        self.eraser_aug_prob = 0.25
        self.eraser_bounds = [20, 300]

        # spatial augmentations
        self.crop_size = crop_size
        self.min_scale = -0.1 # 2^this
        self.max_scale = 1.0 # 2^this
        # self.resize_lim = [0.8, 1.2]
        self.resize_aug_prob = 0.8
        
        self.crop_aug_prob = 0.5
        self.max_crop_offset = 10
        
        self.stretch_prob = 0.8
        self.max_stretch = 0.2
        self.do_flip = True
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.5

        self.switch_dir_prob = 0.3
        # zig zag setup
        self.zigzag_step_max = min(self.S, 5)
        self.zigzag_step_min = 2


    def getitem_helper(self, index):
        sample = None
        gotit = False

        cur_rgb_path = self.rgb_paths[index]
        cur_traj_path = self.traj_paths[index]
        cur_mask_path = self.mask_paths[index]

        start_ind = self.start_inds[index]

        img_names = [folder.split('/')[-1].split('.')[0] for folder in glob.glob(os.path.join(cur_rgb_path, "*"))]
        img_names = sorted(img_names)
        img_names = img_names[start_ind:start_ind+self.S]

        trajs = np.load(os.path.join(cur_traj_path, 'trajs_at_%d.npz' % start_ind), allow_pickle=True)
        trajs = dict(trajs)['trajs'] # S, N 2
        trajs = trajs.astype(np.float32)
        S, N, D = trajs.shape
        assert(S==self.S)
        valids = np.ones((self.S, N)).astype(np.float32)

        if N < self.N:
            return None, False

        rgbs = []
        masks = []
        flows_f = []
        flows_b = []

        for img_name in img_names:
            with Image.open(os.path.join(cur_rgb_path, '{0}.webp'.format(img_name))) as im:
                rgbs.append(np.array(im))
            mask = readImage(os.path.join(cur_mask_path, '{0}.pfm'.format(img_name)))
            masks.append(mask)

        # the data we loaded is all visible
        visibles = np.ones((S, N))

        if self.zigzag or self.S_out > self.S:
            rgbs, masks, trajs, visibles, valids= self.add_zigzag(rgbs, masks, trajs, visibles, valids)

        rgbs, occs, masks, trajs, visibles, valids = self.add_occluders(rgbs, masks, trajs, visibles, valids)

        # print('occ rgbs[0]', rgbs[0].shape)
        if self.use_augs:
            rgbs, trajs, visibles = self.add_photometric_augs(rgbs, trajs, visibles)
            # print('phot rgbs[0]', rgbs[0].shape)
            rgbs, occs, masks, trajs = self.add_spatial_augs(rgbs, occs, masks, trajs)
            # print('spat rgbs[0]', rgbs[0].shape)
        else:
            rgbs, occs, masks, trajs = self.just_crop(rgbs, occs, masks, trajs)
            # print('crop rgbs[0]', rgbs[0].shape)

        # mark oob points as invisible
        for s in range(1, S):
            oob_inds = np.logical_or(np.logical_or(trajs[s,:,0] < 0, trajs[s,:,0] > self.crop_size[1]-1), np.logical_or(trajs[s,:,1] < 0, trajs[s,:,1] > self.crop_size[0]-1))
            visibles[s,oob_inds] = 0

        if self.force_twice_vis:
            # ensure that the point is visible at frame0 and at least one other frame
            vis0 = visibles[0] > 0
            inbound0 = (trajs[0,:,0] >= 0) & (trajs[0,:,0] <= self.crop_size[1]-1) & (trajs[0,:,1] >= 0) & (trajs[0,:,1] <= self.crop_size[0]-1)
            inbound_other = (trajs[1,:,0] >= 0) & (trajs[1,:,0] <= self.crop_size[1]-1) & (trajs[1,:,1] >= 0) & (trajs[1,:,1] <= self.crop_size[0]-1)
            vis_other = visibles[1] > 0
            for s in range(2,S):
                inbound_i = (trajs[s,:,0] >= 0) & (trajs[s,:,0] <= self.crop_size[1]-1) & (trajs[s,:,1] >= 0) & (trajs[s,:,1] <= self.crop_size[0]-1)
                inbound_other = inbound_other | inbound_i
                vis_i = visibles[s] > 0
                vis_other = vis_other | vis_i
            inbound_ok = inbound0 & inbound_other
            vis_ok = vis0 & vis_other
        else:
            assert(False) # only twice inbound is supported right now

        inb_and_vis = inbound_ok & vis_ok
        trajs = trajs[:,inb_and_vis]
        visibles = visibles[:,inb_and_vis]
        valids = valids[:,inb_and_vis]
        
        if trajs.shape[1] <= self.N:
            print('returning None')
            return None, False
        
        N_ = min(trajs.shape[1], self.N)
        
        # inds = utils.py.farthest_point_sample(trajs[0], N_, deterministic=False)
        inds = np.random.choice(trajs.shape[1], N_, replace=False)

        trajs_full = np.zeros((self.S_out, self.N, 2)).astype(np.float32)
        visibles_full = np.zeros((self.S_out, self.N)).astype(np.float32)
        valids_full = np.zeros((self.S_out, self.N)).astype(np.float32)
        # valids = np.zeros((self.N)).astype(np.float32)
        trajs_full[:,:N_] = trajs[:,inds]
        visibles_full[:,:N_] = visibles[:,inds]
        valids_full[:,:N_] = valids[:,inds]

        rgbs = torch.from_numpy(np.stack(rgbs, 0)).permute(0, 3, 1, 2) # S, C, H, W
        occs = torch.from_numpy(np.stack(occs, 0)).unsqueeze(1) # S, 1, H, W
        masks = torch.from_numpy(np.stack(masks, 0)).unsqueeze(1) # S, 1, H, W
        trajs = torch.from_numpy(trajs_full) # S, N, 2
        visibles = torch.from_numpy(visibles_full) # S, N
        valids = torch.from_numpy(valids_full) # S, N

        if torch.sum(valids[0,:]) < self.N_min:
            return None, False

        sample = {
            'rgbs': rgbs,
            'occs': occs,
            'masks': masks,
            'trajs': trajs,
            'visibles': visibles,
            'valids': valids,
        }
        return sample, True
    
    def __getitem__(self, index):
        gotit = False
        
        while not gotit:
            sample, gotit = self.getitem_helper(index)

            if not gotit:
                # save time by not trying this index again
                load_fail = 1
                self.load_fails[index] = load_fail
                print('updated load_fails (on this worker): %d/%d...' % (np.sum(self.load_fails), len(self.load_fails)))

                while load_fail:
                    index = np.random.randint(0, len(self.load_fails))
                    load_fail = self.load_fails[index]

        return sample

    def add_occluders(self, rgbs, masks, trajs, visibles, valids):
        '''
        Input:
            rgbs --- list of len S, each = np.array (H, W, 3)
            trajs --- np.array (S, N, 2)
        Output:
            rgbs_aug --- np.array (S, H, W, 3)
            trajs_aug --- np.array (S, N_new, 2)
            visibles_aug --- np.array (S, N_new)
        '''

        T, N, _ = trajs.shape

        # print('trajs', trajs.shape)
        # print('len(rgbs)', len(rgbs))
        
        S = len(rgbs)
        H, W = rgbs[0].shape[:2]

        assert(S==T)

        # rgbs = [0.1*rgb.astype(np.float32) for rgb in rgbs]
        rgbs = [rgb.astype(np.float32) for rgb in rgbs]
        occs = [np.zeros_like(rgb[:,:,0]) for rgb in rgbs]

        max_occ = 10
        alt_inds = np.random.choice(len(self.occ_rgb_paths), max_occ, replace=False)
        
        ############ occluders from other videos ############
        for oi in range(max_occ): # number of occluders:
            # alt_ind = np.random.choice(len(self.occ_rgb_paths))
            alt_ind = alt_inds[oi]
            occ_rgb_path = self.occ_rgb_paths[alt_ind]
            occ_mask_path = self.occ_mask_paths[alt_ind]
            occ_start_ind = self.occ_start_inds[alt_ind]
            occ_traj_path = self.occ_traj_paths[alt_ind]

            # print('occ_rgb_path', occ_rgb_path)
            # print('occ_start_ind', occ_start_ind)
            # print('occ_traj_path', occ_traj_path)
            
            img_names = [folder.split('/')[-1].split('.')[0] for folder in glob.glob(os.path.join(occ_rgb_path, "*"))]
            img_names = sorted(img_names)
            img_names = img_names[occ_start_ind:occ_start_ind+self.S]

            occ_info = np.load(occ_traj_path, allow_pickle=True).item()
            id_str = list(occ_info.keys())[np.random.choice(len(occ_info))]
            alt_trajs = occ_info[id_str] # S,N,2, with often N==0

            occ_id = int(id_str)

            alt_rgbs = []
            alt_masks = []
            alt_masks_blur = []

            for img_name in img_names:
                with Image.open(os.path.join(occ_rgb_path, '{0}.webp'.format(img_name))) as im:
                    alt_rgbs.append(np.array(im))
                mask = readImage(os.path.join(occ_mask_path, '{0}.pfm'.format(img_name)))
                mask = (mask==occ_id).astype(np.float32)
                # mask_  = np.clip(cv2.GaussianBlur(mask,(3,3),0) + mask, 0,1).reshape(H, W, 1) # widen slightly, but keep all the important pixels
                mask_blur = np.clip(cv2.GaussianBlur(mask,(3,3),0), 0,1).reshape(H, W, 1)
                alt_masks.append(mask)#.reshape(H, W, 1))
                alt_masks_blur.append(mask_blur)#.reshape(H, W, 1))

            '''
            check whether this occluder is legal
            '''

            legal_thres = 0.2

            try:
                if np.sum(alt_masks[0]) / np.sum(alt_masks[1]) < legal_thres \
                        or np.sum(alt_masks[-1]) / np.sum(alt_masks[-2]) < legal_thres:
                    continue
            except ZeroDivisionError:
                continue

            '''
            try to make occluders appear randomly at different timestep
            however, in this implementation it's hard to detect illegal appearance & disappearance
            and trajs will be noisy, as some points on the occluders can never be tracked during the disappearance time
            '''

            # # set visibles = 0 and change it to 1 only the occluders are added to the frame
            # alt_visibles = np.zeros((self.S_out, alt_trajs.shape[1]))  # S,N
            # alt_valids = np.ones((self.S_out, alt_trajs.shape[1]))  # S,N
            #
            # # if the occluder is right, we decide when it should appear and how many times it appear
            # alt_rgbs_extend = []
            # alt_masks_extend = []
            # alt_masks_blur_extend = []
            #
            # # repeat times and when they begin
            # # occ_repeat_num = 1 means the occluder only shows up once
            # occ_repeat_num = int(np.random.randint(low=1, high=int(self.S_out // self.S), size=(1, )))
            # occ_new_start_ind = np.random.choice(self.S_out, occ_repeat_num, replace=False)
            #
            # blank_frame = np.zeros((H, W))
            # blank_frame_3d = np.zeros((H, W, 3))
            # blank_traj = -1 * np.ones((1, alt_trajs.shape[1], 2))
            #
            # alt_trajs_extend = []
            #
            # for repeat_idx in range(occ_repeat_num):
            #
            #     alt_visibles[occ_new_start_ind[repeat_idx]:occ_new_start_ind[repeat_idx]+self.S] = 1
            #
            #     alt_rgbs_extend.extend([blank_frame_3d for _ in range(occ_new_start_ind[repeat_idx])])
            #     alt_rgbs_extend.extend(alt_rgbs)
            #
            #     alt_masks_extend.extend([blank_frame for _ in range(occ_new_start_ind[repeat_idx])])
            #     alt_masks_extend.extend(alt_masks)
            #
            #     alt_masks_blur_extend.extend([blank_frame for _ in range(occ_new_start_ind[repeat_idx])])
            #     alt_masks_blur_extend.extend(alt_masks_blur)
            #
            #     alt_trajs_extend.extend([blank_traj for _ in range(occ_new_start_ind[repeat_idx])])
            #     alt_trajs_extend.append(alt_trajs)
            #
            # padding_frame_num = self.S_out - len(alt_masks_extend)
            #
            # if padding_frame_num > 0:
            #     alt_rgbs_extend.extend([blank_frame_3d for _ in range(padding_frame_num)])
            #     alt_masks_extend.extend([blank_frame for _ in range(padding_frame_num)])
            #     alt_masks_blur_extend.extend([blank_frame for _ in range(padding_frame_num)])
            #     alt_trajs_extend.extend([blank_traj for _ in range(padding_frame_num)])
            #
            # alt_rgbs = alt_rgbs_extend[:self.S_out]
            # alt_masks = alt_masks_extend[:self.S_out]
            # alt_masks_blur = alt_masks_blur_extend[:self.S_out]
            #
            # alt_trajs_extend = np.concatenate(alt_trajs_extend, axis=0)
            # alt_trajs = alt_trajs_extend[:self.S_out]

            '''
            so i use simple forward & backward movement for the occluders
            '''

            alt_rgbs_extend = []
            alt_masks_extend = []
            alt_masks_blur_extend = []
            alt_trajs_extend = []

            alt_visibles = np.ones((self.S_out, alt_trajs.shape[1]))  # S,N
            alt_valids = np.ones((self.S_out, alt_trajs.shape[1]))  # S,N

            for _ in range(self.S_out // self.S):
                alt_rgbs_extend.extend(alt_rgbs)
                alt_masks_extend.extend(alt_masks)
                alt_masks_blur_extend.extend(alt_masks_blur)
                alt_trajs_extend.extend(alt_trajs)

                alt_rgbs.reverse()
                alt_masks_extend.reverse()
                alt_masks_blur_extend.reverse()
                alt_trajs = alt_trajs[::-1]

            alt_rgbs = alt_rgbs_extend
            alt_masks = alt_masks_extend
            alt_masks_blur = alt_masks_blur_extend
            alt_trajs = np.stack(alt_trajs_extend, axis=0)

            '''we can test random appearing version by commenting codes up there '''

            rgbs = [rgb*(1.0-alt_mask.reshape(H,W,1))+alt_rgb*alt_mask.reshape(H,W,1) for (rgb,alt_rgb,alt_mask) in zip(rgbs,alt_rgbs,alt_masks_blur)]
            # occs = [np.clip(occ+alt_mask, 0,1) for (occ,alt_mask) in zip(occs,alt_masks)]

            occs = [occ+alt_mask for (occ, alt_mask) in zip(occs, alt_masks)]

            # # darken the non-occluder, for debug
            # rgbs = [rgb*(1.0-(alt_mask*0.5)) for (rgb,alt_rgb,alt_mask) in zip(rgbs,alt_rgbs,alt_masks)]

            # any prev traj in the new masks should be marked invisible
            for s in range(S):
                xy = trajs[s].round().astype(np.int32) # N, 2
                x, y = xy[:,0], xy[:,1] # N
                # cond1 = (x >= 0) & (x <= W-1) & (y >= 0) & (y <= H-1)
                # x = x[inds]
                # y = [inds]
                x_ = x.clip(0, W-1)
                y_ = y.clip(0, H-1)
                inds = (alt_masks[s][y_,x_] == 1) & (x >= 0) & (x <= W-1) & (y >= 0) & (y <= H-1)
                # inds = np.logical_and(np.logical_and( >= x0, trajs[i,:,0] < x1), np.logical_and(trajs[i,:,1] >= y0, trajs[i,:,1] < y1))
                visibles[s, inds] = 0

            rgbs = [rgb.astype(np.uint8) for rgb in rgbs]
            trajs = np.concatenate([trajs, alt_trajs], axis=1)
            visibles = np.concatenate([visibles, alt_visibles], axis=1)
            valids = np.concatenate([valids, alt_valids], axis=1)

        return rgbs, occs, masks, trajs, visibles, valids

    def add_zigzag(self, rgbs, masks, trajs, visibles, valids):
        '''
               Input:
                   rgbs --- list of len S, each = np.array (H, W, 3)
                   trajs --- np.array (S, N, 2)
               Output:
                   rgbs --- np.array (S_out, H, W, 3)
                   trajs --- np.array (S_out, N_new,  2)
                   visibles --- np.array (S_out, N_new)
               '''

        T, N, _ = trajs.shape

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]

        ############ create a zig zag sequence ############=
        rgbs_zigzag = []
        masks_zigzag = []
        trajs_zigzag = []
        visibles_zigzag = []
        valids_zigzag = []
        cur_index = 0
        direction = 1

        while len(rgbs_zigzag) < self.S_out:
            # steps to go
            num_steps = np.random.randint(low=self.zigzag_step_min, high=self.zigzag_step_max + 1)
            num_steps = min(num_steps, self.S_out - len(rgbs_zigzag))

            # whether to change direction
            if np.random.rand() < self.switch_dir_prob:
                direction *= -1

            # we start at the current location (inclusive) and add frames sequentially
            for _ in range(num_steps):
                rgbs_zigzag.append(rgbs[cur_index])
                masks_zigzag.append(masks[cur_index])
                trajs_zigzag.append(trajs[cur_index])
                visibles_zigzag.append(visibles[cur_index])
                valids_zigzag.append(valids[cur_index])

                cur_index += direction

                # if this got too small (means we were at 0, and now at -1), let's bounce back the direction
                # and set cur_index to 1
                if cur_index == -1:
                    direction = 1
                    cur_index = 1

                # if this got too large (means we were at S-1, and now at S), let's bounce back the direction
                # and set cur_index to S-2
                if cur_index == S:
                    direction = -1
                    cur_index = S - 2

        rgbs = rgbs_zigzag
        masks = masks_zigzag
        trajs = np.stack(trajs_zigzag, axis=0)
        visibles = np.stack(visibles_zigzag, axis=0)
        valids = np.stack(valids_zigzag, axis=0)

        return rgbs, masks, trajs, visibles, valids

    def add_photometric_augs(self, rgbs, trajs, visibles):
        T, N, _ = trajs.shape

        # print('trajs', trajs.shape)
        # print('len(rgbs)', len(rgbs))
        
        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert(S==T)

        # rgbs = [0.1*rgb.astype(np.float32) for rgb in rgbs]
        
        ############ eraser transform (per image after the first) ############
        rgbs = [rgb.astype(np.float32) for rgb in rgbs]
        for i in range(1, S):
            if np.random.rand() < self.eraser_aug_prob:
                mean_color = np.mean(rgbs[i].reshape(-1, 3), axis=0)
                for _ in range(np.random.randint(1, 3)): # number of times to occlude
                    xc = np.random.randint(0, W)
                    yc = np.random.randint(0, H)
                    dx = np.random.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                    dy = np.random.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                    x0 = np.clip(xc - dx/2, 0, W-1).round().astype(np.int32)
                    x1 = np.clip(xc + dx/2, 0, W-1).round().astype(np.int32)
                    y0 = np.clip(yc - dy/2, 0, W-1).round().astype(np.int32)
                    y1 = np.clip(yc + dy/2, 0, W-1).round().astype(np.int32)
                    # print(x0, x1, y0, y1)
                    rgbs[i][y0:y1, x0:x1, :] = mean_color

                    occ_inds = np.logical_and(np.logical_and(trajs[i,:,0] >= x0, trajs[i,:,0] < x1), np.logical_and(trajs[i,:,1] >= y0, trajs[i,:,1] < y1))
                    visibles[i, occ_inds] = 0
        rgbs = [rgb.astype(np.uint8) for rgb in rgbs]

        ############ photometric augmentation ############
        if np.random.rand() < self.color_aug_prob:
            # random per-frame amount of aug
            rgbs = [np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]

        if np.random.rand() < self.blur_aug_prob:
            # random per-frame amount of blur
            rgbs = [np.array(self.blur_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]

        return rgbs, trajs, visibles

    def add_spatial_augs(self, rgbs, occs, masks, trajs):
        T, N, _ = trajs.shape

        # print('trajs', trajs.shape)
        # print('len(rgbs)', len(rgbs))
        
        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert(S==T)

        rgbs = [rgb.astype(np.float32) for rgb in rgbs]
        
        ############ spatial transform ############

        # scaling + stretching
        scale_x = 1.0
        scale_y = 1.0
        H_new = H
        W_new = W
        if np.random.rand() < self.resize_aug_prob:
            # print('spat')
            min_scale = np.maximum(
                (self.crop_size[0] + 8) / float(H),
                (self.crop_size[1] + 8) / float(W))

            scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
            scale_x = scale 
            scale_y = scale
            # print('scale', scale)

            if np.random.rand() < self.stretch_prob:
                # print('stretch')
                scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
                scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

            scale_x = np.clip(scale_x, min_scale, None)
            scale_y = np.clip(scale_y, min_scale, None)

            # print('scale_x,y', scale_x, scale_y)

            H_new = int(H * scale_y)
            W_new = int(W * scale_x)

            # print('H_new, W_new', H_new, W_new)
            # dim_resize = (W_new, H_new * S)
            rgbs = [cv2.resize(rgb, (W_new, H_new), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
            occs = [cv2.resize(occ, (W_new, H_new), interpolation=cv2.INTER_LINEAR) for occ in occs]
            masks = [cv2.resize(mask, (W_new, H_new), interpolation=cv2.INTER_LINEAR) for mask in masks]
        trajs[:,:,0] *= scale_x
        trajs[:,:,1] *= scale_y
        
        if np.random.rand() < self.crop_aug_prob:
            # per-timestep crop
            y0 = np.random.randint(0, H_new - self.crop_size[0])
            x0 = np.random.randint(0, W_new - self.crop_size[1])
            for s in range(S):
                # on each frame, maybe shift a bit more 
                if s > 0 and np.random.rand() < self.crop_aug_prob:
                    x0 = x0 + np.random.randint(-self.max_crop_offset, self.max_crop_offset+1)
                    y0 = y0 + np.random.randint(-self.max_crop_offset, self.max_crop_offset+1)
                y0 = min(max(0, y0), H_new - self.crop_size[0] - 1)
                x0 = min(max(0, x0), W_new - self.crop_size[1] - 1)
                rgbs[s] = rgbs[s][y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
                occs[s] = occs[s][y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
                masks[s] = masks[s][y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
                trajs[s,:,0] -= x0
                trajs[s,:,1] -= y0
        else:
            # simple crop
            y0 = np.random.randint(0, H_new - self.crop_size[0])
            x0 = np.random.randint(0, W_new - self.crop_size[1])
            rgbs = [rgb[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]] for rgb in rgbs]
            occs = [occ[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]] for occ in occs]
            masks = [mask[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]] for mask in masks]
            trajs[:,:,0] -= x0
            trajs[:,:,1] -= y0
            
        H_new = self.crop_size[0]
        W_new = self.crop_size[1]

        # flip
        h_flipped = False
        v_flipped = False
        if self.do_flip:
            # h flip
            if np.random.rand() < self.h_flip_prob:
                # print('h flip')
                h_flipped = True
                rgbs = [rgb[:,::-1] for rgb in rgbs]
                occs = [occ[:,::-1] for occ in occs]
                masks = [mask[:,::-1] for mask in masks]
            # v flip
            if np.random.rand() < self.v_flip_prob:
                # print('v flip')
                v_flipped = True
                rgbs = [rgb[::-1] for rgb in rgbs]
                occs = [occ[::-1] for occ in occs]
                masks = [mask[::-1] for mask in masks]
        if h_flipped:
            trajs[:,:,0] = W_new - trajs[:,:,0]
        if v_flipped:
            trajs[:,:,1] = H_new - trajs[:,:,1]
            
        return rgbs, occs, masks, trajs

    def just_crop(self, rgbs, occs, masks, trajs):
        T, N, _ = trajs.shape
        
        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert(S==T)

        ############ spatial transform ############

        H_new = H
        W_new = W

        # simple random crop
        y0 = np.random.randint(0, H_new - self.crop_size[0])
        x0 = np.random.randint(0, W_new - self.crop_size[1])
        rgbs = [rgb[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]] for rgb in rgbs]
        occs = [occ[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]] for occ in occs]
        masks = [mask[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]] for mask in masks]
        trajs[:,:,0] -= x0
        trajs[:,:,1] -= y0
            
        return rgbs, occs, masks, trajs
    
    def __len__(self):
        # return 10
        return len(self.rgb_paths)


class FlyingThingsZigzagDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_location='../../aharley/flyingthings', dset='TRAIN', subset='all', use_augs=False, N=0, S=8,
                 S_out=32, crop_size=(368, 496), version='ad', occ_version='al', force_double_inb=False, force_all_inb=False):

        self.S = S
        self.S_out = S_out
        self.N = N

        self.use_augs = use_augs

        self.force_double_inb = force_double_inb
        self.force_all_inb = force_all_inb

        self.rgb_paths = []
        self.traj_paths = []
        self.mask_paths = []
        self.flow_f_paths = []
        self.flow_b_paths = []
        self.start_inds = []
        self.load_fails = []

        self.subset = subset

        if self.subset == 'all':
            subsets = ['A', 'B', 'C']
        else:
            subsets = [subset]

        for subset in subsets:
            rgb_root_path = os.path.join(dataset_location, "frames_cleanpass_webp", dset, subset)
            flow_root_path = os.path.join(dataset_location, "optical_flow", dset, subset)
            traj_root_path = os.path.join(dataset_location, "trajs_%s" % version, dset, subset)
            mask_root_path = os.path.join(dataset_location, "object_index", dset, subset)
            # heavy_root_path = os.path.join(dataset_location, "heavy_raft_flows_ab", dset, subset)

            folder_names = [folder.split('/')[-1] for folder in glob.glob(os.path.join(traj_root_path, "*"))]
            folder_names = sorted(folder_names)
            # print('first 10 folders only')
            # folder_names = folder_names[:10]
            # print('folder_names', folder_names)

            for ii, folder_name in enumerate(folder_names):
                for lr in ['left', 'right']:
                    cur_rgb_path = os.path.join(rgb_root_path, folder_name, lr)
                    cur_traj_path = os.path.join(traj_root_path, folder_name, lr)
                    cur_mask_path = os.path.join(mask_root_path, folder_name, lr)

                    for start_ind in [0, 1, 2, 3]:
                        traj_fn = cur_traj_path + '/trajs_at_%d.npz' % start_ind
                        if os.path.isfile(traj_fn):
                            file_size = os.path.getsize(traj_fn)
                            if file_size > 1000:  # the empty ones are 264 bytes
                                # trajs = np.load(os.path.join(cur_traj_path, 'trajs_at_%d.npz' % start_ind), allow_pickle=True)
                                # trajs = dict(trajs)['trajs'] # S,N,2
                                # S, N, D = trajs.shape
                                # if N >= self.N:
                                #     # print('adding this one')
                                #     self.rgb_paths.append(cur_rgb_path)
                                #     self.traj_paths.append(cur_traj_path)
                                #     self.mask_paths.append(cur_mask_path)
                                #     self.start_inds.append(start_ind)
                                #     self.load_fails.append(0)
                                #     sys.stdout.write('.')
                                #     sys.stdout.flush()
                                # else:
                                #     sys.stdout.write('l')
                                #     sys.stdout.flush()

                                self.rgb_paths.append(cur_rgb_path)
                                self.traj_paths.append(cur_traj_path)
                                self.mask_paths.append(cur_mask_path)
                                self.start_inds.append(start_ind)
                                self.load_fails.append(0)
                                sys.stdout.write('.')
                                sys.stdout.flush()

                # if ii > 10 and (ii % 100)==0:
                #     # print('found %d samples in %s so far...' % (len(self.rgb_paths), dataset_location))
                #     sys.stdout.write('%.1f%%' % (100*(ii+1)/len(folder_names)))
                #     sys.stdout.flush()
        # print('%.1f%%' % (100*(ii+1)/len(folder_names)))
        print('found %d samples in %s (dset=%s, subset=%s, version=%s)' % (
        len(self.rgb_paths), dataset_location, dset, self.subset, version))

        # we also need to step through and collect ooccluder info
        print('loading occluders...')

        self.occ_rgb_paths = []
        self.occ_mask_paths = []
        self.occ_start_inds = []
        self.occ_traj_paths = []

        # print('locking start_ind=0, for speed')
        for subset in subsets:
            # print('sub')
            # print(subset*10)

            rgb_root_path = os.path.join(dataset_location, "frames_cleanpass_webp", dset, subset)
            flow_root_path = os.path.join(dataset_location, "optical_flow", dset, subset)
            mask_root_path = os.path.join(dataset_location, "object_index", dset, subset)
            occ_root_path = os.path.join(dataset_location, "occluders_%s" % occ_version, dset, subset)

            # print('occ_root_path', occ_root_path)

            folder_names = [folder.split('/')[-1] for folder in glob.glob(os.path.join(occ_root_path, "*"))]
            folder_names = sorted(folder_names)
            # print('folder_names', folder_names)
            # print('first 10 folders only')
            # folder_names = folder_names[:10]

            # print('folder_names', folder_names)

            # for ii, folder_name in enumerate(folder_names):
            for folder_name in folder_names:

                for lr in ['left', 'right']:

                    cur_rgb_path = os.path.join(rgb_root_path, folder_name, lr)
                    cur_mask_path = os.path.join(mask_root_path, folder_name, lr)
                    cur_occ_path = os.path.join(occ_root_path, folder_name, lr)

                    # start_ind = 0
                    # if True:
                    for start_ind in [0, 1, 2]:
                        occ_fn = cur_occ_path + '/occluder_at_%d.npy' % (start_ind)

                        # print("occ_fn', occ_fn')
                        # # file_names = glob.glob(os.path.join(cur_occ_path, "*at_%d.npz" % start_ind))
                        # file_names = glob.glob(os.path.join(cur_occ_path, "*at_%d.npy" % start_ind))
                        # print('file_names', file_names)
                        # print('len', len(file_names))
                        # # input()

                        # for ii in range(len(file_names)):
                        if os.path.isfile(occ_fn):
                            file_size = os.path.getsize(occ_fn)
                            if file_size > 1000:  # the empty ones are 10 bytes

                                # print('file_size', file_size)

                                # occ_info = np.load(occ_fn, allow_pickle=True)
                                # occ_info = dict(occ_info)
                                # occ_trajs = occ_info['trajs'] # S,N,2, or None
                                # occ_id = occ_info['id_'] # []
                                # print('occ_trajs', occ_trajs.shape)

                                self.occ_rgb_paths.append(cur_rgb_path)
                                self.occ_mask_paths.append(cur_mask_path)
                                self.occ_start_inds.append(start_ind)
                                self.occ_traj_paths.append(occ_fn)
                                # print('adding something')

                        sys.stdout.write('.')
                        sys.stdout.flush()

                # if ii > 10 and (ii % 100)==0:
                #     # print('found %d samples in %s so far...' % (len(self.rgb_paths), dataset_location))
                #     sys.stdout.write('%.1f%%' % (100*(ii+1)/len(folder_names)))
                #     sys.stdout.flush()
        # print('%.1f%%' % (100*(ii+1)/len(folder_names)))
        print('found %d occluders in %s (dset=%s, subset=%s, version=%s)' % (
        len(self.occ_rgb_paths), dataset_location, dset, self.subset, occ_version))

        # print('self.occ_rgb_paths', self.occ_rgb_paths[:5])
        # print('self.occ_traj_paths', self.occ_traj_paths[:5])

        # photometric augmentation
        # self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.photo_aug = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.25 / 3.14)
        self.blur_aug = GaussianBlur(11, sigma=(0.1, 2.0))
        self.asymmetric_color_aug_prob = 0.2

        self.blur_aug_prob = 0.2
        self.color_aug_prob = 0.5

        # occlusion augmentation
        self.eraser_aug_prob = 0.1
        self.eraser_bounds = [20, 300]

        # spatial augmentations
        self.crop_size = crop_size
        self.min_scale = -0.1  # 2^this
        self.max_scale = 1.0  # 2^this
        # self.resize_lim = [0.8, 1.2]
        self.resize_aug_prob = 0.8

        # spatial augmentations
        self.crop_aug_prob = 0.5
        self.max_crop_offset = 10

        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.2
        self.max_stretch = 0.2
        self.do_flip = False
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.5

        # zig zag setup
        self.zigzag_step_max = min(self.S, 5)
        self.zigzag_step_min = 2
        self.switch_dir_prob = 0
        self.max_crop_offset = 20

    def __getitem__(self, index):
        gotit = False
        while not gotit:
            cur_rgb_path = self.rgb_paths[index]
            cur_traj_path = self.traj_paths[index]
            cur_mask_path = self.mask_paths[index]
            # cur_heavy_path = self.heavy_paths[index]
            # cur_flow_f_path = self.flow_f_paths[index]
            # cur_flow_b_path = self.flow_b_paths[index]
            # print('cur_rgb_path', cur_rgb_path)

            img_names = [folder.split('/')[-1].split('.')[0] for folder in glob.glob(os.path.join(cur_rgb_path, "*"))]
            img_names = sorted(img_names)

            start_ind = 0
            img_names = img_names[start_ind:start_ind + self.S]

            rgbs = []
            masks = []
            flows_f = []
            flows_b = []

            # orig_rgbs = []
            # orig_trajs = []

            for img_name in img_names:
                # im = Image.open(os.path.join(cur_rgb_path, '{0}.webp'.format(img_name))) # H, W, 3
                with Image.open(os.path.join(cur_rgb_path, '{0}.webp'.format(img_name))) as im:
                    rgbs.append(np.array(im))

                # print('rgb', rgbs[-1].shape)
                # print('cur_mask_path', cur_mask_path)
                mask = readImage(os.path.join(cur_mask_path, '{0}.pfm'.format(img_name)))
                # print('%s mask' % img_name, mask.shape)
                masks.append(mask)

            trajs = np.load(os.path.join(cur_traj_path, 'trajs_at_%d.npz' % start_ind), allow_pickle=True)
            trajs = dict(trajs)['trajs']  # S, N, 2

            # heavy = np.load(os.path.join(cur_heavy_path, 'traj.npz'), allow_pickle=True)
            # heavy = dict(heavy)
            # trajs_XYs = heavy['trajs_XYs']
            # trajs_Ts = heavy['trajs_Ts']

            # print('trajs_XYs[0]', trajs_XYs[0], trajs_XYs[0].shape)
            # print('trajs_Ts[0]', trajs_Ts[0], trajs_Ts[0].shape)

            try:
                S_, N_, D_ = trajs.shape
            except ValueError:
                index += 1
                continue

            assert (N_ >= self.N)
            assert (S_ >= self.S)

            trajs = trajs[:self.S, :]
            # print('trajs', trajs.shape)

            orig_rgbs = torch.from_numpy(np.stack(rgbs, 0)).permute(0, 3, 1, 2).clone()  # S, C, H, W
            orig_trajs = torch.from_numpy(trajs)  # S, N, 2

            if self.use_augs:
                success = False
                for _ in range(5):
                    rgbs_aug, masks_aug, trajs_aug, visibles_aug, inbound = self.zigzag_augment(rgbs, masks, trajs)

                    try:
                        N_ = trajs_aug.shape[1]
                        if N_ >= self.N:
                            success = True
                        break
                    except:
                        continue
                rgbs, masks, trajs, visibles = rgbs_aug, masks_aug, trajs_aug, visibles_aug
                if trajs.shape[1] < self.N:
                    return self.__getitem__(np.random.choice(self.__len__()))
            else:
                for _ in range(5):
                    rgbs_crop, masks_crop, trajs_crop, visibles_crop, inbound = self.zigzag_just_crop(rgbs, masks,
                                                                                                      trajs)
                    try:
                        N_ = trajs_crop.shape[1]
                        if N_ >= self.N:
                            success = True
                        break
                    except:
                        continue
                rgbs, masks, trajs, visibles = rgbs_crop, masks_crop, trajs_crop, visibles_crop

            orig_trajs = orig_trajs[:, inbound]

            N_ = min(trajs.shape[1], self.N)
            traj_id = np.random.choice(trajs.shape[1], size=N_, replace=False)
            # trajs = trajs[traj_id] # N, S, 2

            trajs_full = np.zeros((self.S_out, self.N, 2)).astype(np.float32)
            valids = np.zeros((self.N)).astype(np.float32)
            trajs_full[:, :N_] = trajs[:, traj_id]
            valids[:N_] = 1
            visibles = visibles[:, traj_id]

            orig_trajs = orig_trajs[:, traj_id]

            rgbs = torch.from_numpy(np.stack(rgbs, 0)).permute(0, 3, 1, 2)  # S, C, H, W
            masks = torch.from_numpy(np.stack(masks, 0)).unsqueeze(1)  # S, 1, H, W
            trajs = torch.from_numpy(trajs_full)  # S, N, 2
            valids = torch.from_numpy(valids)  # N
            visibles = torch.from_numpy(visibles)   # S, N

            if torch.sum(valids) == self.N:
                gotit = True
            else:
                # print('re-indexing...')
                index = np.random.randint(0, len(self.rgb_paths))

        # print('torch.sum(valids)', torch.sum(valids))
        return_dict = {
            # 'cur_heavy_path': cur_heavy_path,
            'orig_rgbs': orig_rgbs,
            'orig_trajs': orig_trajs,
            'rgbs': rgbs,
            'masks': masks,
            'trajs': trajs,
            # 'trajs_XYs': [trajs_XYs],
            # 'trajs_Ts': [trajs_Ts],
            'valids': valids,
            'visibles': visibles,
        }
        return return_dict

    def zigzag_augment(self, rgbs, masks, trajs):
        '''
        Input:
            rgbs --- list of len S, each = np.array (H, W, 3)
            trajs --- np.array (S, N, 2)
        Output:
            rgbs_aug --- np.array (S_out, H, W, 3)
            trajs_aug --- np.array (S_out, N_new, 2)
            visibles_aug --- np.array (S_out, N_new)
        '''

        T, N, _ = trajs.shape
        visibles = np.ones((T, N))

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]

        ############ create a zig zag sequence ############=
        rgbs_zigzag = []
        masks_zigzag = []
        trajs_zigzag = []
        cur_index = 0
        direction = 1

        while len(rgbs_zigzag) < self.S_out:
            # steps to go
            num_steps = np.random.randint(low=self.zigzag_step_min, high=self.zigzag_step_max + 1)
            num_steps = min(num_steps, self.S_out - len(rgbs_zigzag))

            # whether to change direction
            if np.random.rand() < self.switch_dir_prob:
                direction *= -1

            # we start at the current location (inclusive) and add frames sequentially
            for _ in range(num_steps):
                rgbs_zigzag.append(rgbs[cur_index])
                masks_zigzag.append(masks[cur_index])
                trajs_zigzag.append(trajs[cur_index])

                cur_index += direction

                # if this got too small (means we were at 0, and now at -1), let's bounce back the direction
                # and set cur_index to 1
                if cur_index == -1:
                    direction = 1
                    cur_index = 1

                # if this got too large (means we were at S-1, and now at S), let's bounce back the direction
                # and set cur_index to S-2
                if cur_index == S:
                    direction = -1
                    cur_index = S - 2

        rgbs = rgbs_zigzag
        masks = masks_zigzag
        trajs = np.stack(trajs_zigzag, axis=0)

        T, N, _ = trajs.shape
        visibles = np.ones((T, N))
        S = len(rgbs)
        H, W = rgbs[0].shape[:2]

        ############ photometric augmentation ############
        if np.random.rand() < self.asymmetric_color_aug_prob:
            for i in range(S):
                rgbs[i] = np.array(self.photo_aug(Image.fromarray(rgbs[i])), dtype=np.uint8)
        else:
            rgbs = [np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]

        ############ eraser transform (per image) ############
        for i in range(1, S):
            if np.random.rand() < self.eraser_aug_prob:
                mean_color = np.mean(rgbs[i].reshape(-1, 3), axis=0)
                for _ in range(np.random.randint(1, 6)):
                    xc = np.random.randint(0, W)
                    yc = np.random.randint(0, H)
                    dx = np.random.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                    dy = np.random.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                    x0 = np.clip(xc - dx / 2, 0, W - 1).round().astype(np.int32)
                    x1 = np.clip(xc + dx / 2, 0, W - 1).round().astype(np.int32)
                    y0 = np.clip(yc - dy / 2, 0, W - 1).round().astype(np.int32)
                    y1 = np.clip(yc + dy / 2, 0, W - 1).round().astype(np.int32)
                    # print(x0, x1, y0, y1)
                    rgbs[i][y0:y1, x0:x1, :] = mean_color

                    occ_inds = np.logical_and(np.logical_and(trajs[i, :, 0] >= x0, trajs[i, :, 0] < x1),
                                              np.logical_and(trajs[i, :, 1] >= y0, trajs[i, :, 1] < y1))
                    visibles[i, occ_inds] = 0

        ############ spatial transform ############
        # scaling + stretching
        scale_x = 1.0
        scale_y = 1.0
        H_new = H
        W_new = W
        if np.random.rand() < self.spatial_aug_prob:
            # print('spat')
            min_scale = np.maximum(
                (self.crop_size[0] + 8) / float(H),
                (self.crop_size[1] + 8) / float(W))

            scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
            scale_x = scale
            scale_y = scale

            if np.random.rand() < self.stretch_prob:
                # print('stretch')
                scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
                scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

            scale_x = np.clip(scale_x, min_scale, None)
            scale_y = np.clip(scale_y, min_scale, None)

            H_new = int(H * scale_y)
            W_new = int(W * scale_x)
            # dim_resize = (W_new, H_new * S)
            rgbs = [cv2.resize(rgb, (W_new, H_new), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
            masks = [cv2.resize(mask, (W_new, H_new), interpolation=cv2.INTER_LINEAR) for mask in masks]

        trajs[:, :, 0] *= scale_x
        trajs[:, :, 1] *= scale_y

        # flip
        h_flipped = False
        v_flipped = False
        if self.do_flip:
            # h flip
            if np.random.rand() < self.h_flip_prob:
                # print('h flip')
                h_flipped = True
                rgbs = [rgb[:, ::-1] for rgb in rgbs]
                masks = [mask[:, ::-1] for mask in masks]
            # v flip
            if np.random.rand() < self.v_flip_prob:
                # print('v flip')
                v_flipped = True
                rgbs = [rgb[::-1] for rgb in rgbs]
                masks = [mask[::-1] for mask in masks]
        if h_flipped:
            trajs[:, :, 0] = W_new - trajs[:, :, 0]
        if v_flipped:
            trajs[:, :, 1] = H_new - trajs[:, :, 1]

        y0 = np.random.randint(0, H_new - self.crop_size[0])
        x0 = np.random.randint(0, W_new - self.crop_size[1])
        for s in range(S):
            if s > 0:
                x0 = x0 + np.random.randint(-self.max_crop_offset, self.max_crop_offset + 1)
                y0 = y0 + np.random.randint(-self.max_crop_offset, self.max_crop_offset + 1)
            y0 = min(max(0, y0), H_new - self.crop_size[0] - 1)
            x0 = min(max(0, x0), W_new - self.crop_size[1] - 1)
            rgbs[s] = rgbs[s][y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            masks[s] = masks[s][y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
            trajs[s, :, 0] -= x0
            trajs[s, :, 1] -= y0
        # print('-')

        if self.force_double_inb:
            inbound0 = (trajs[0, :, 0] >= 0) & (trajs[0, :, 0] <= self.crop_size[1] - 1) & (trajs[0, :, 1] >= 0) & (
                        trajs[0, :, 1] <= self.crop_size[0] - 1)
            inbound1 = (trajs[1, :, 0] >= 0) & (trajs[1, :, 0] <= self.crop_size[1] - 1) & (trajs[1, :, 1] >= 0) & (
                        trajs[1, :, 1] <= self.crop_size[0] - 1)
            inbound = inbound0 & inbound1
        elif self.force_all_inb:
            inbound = (trajs[0, :, 0] >= 0) & (trajs[0, :, 0] <= self.crop_size[1] - 1) & (trajs[0, :, 1] >= 0) & (
                        trajs[0, :, 1] <= self.crop_size[0] - 1)
            for s in range(1, S):
                inboundi = (trajs[s, :, 0] >= 0) & (trajs[s, :, 0] <= self.crop_size[1] - 1) & (trajs[s, :, 1] >= 0) & (
                        trajs[s, :, 1] <= self.crop_size[0] - 1)
                inbound = inbound & inboundi
        else:
            inbound = (trajs[0, :, 0] >= 0) & (trajs[0, :, 0] <= self.crop_size[1] - 1) & (trajs[0, :, 1] >= 0) & (
                        trajs[0, :, 1] <= self.crop_size[0] - 1)

        trajs = trajs[:, inbound]
        visibles = visibles[:, inbound]
        # mark oob points as invisible
        for i in range(1, S):
            oob_inds = np.logical_or(np.logical_or(trajs[i, :, 0] < 0, trajs[i, :, 0] > self.crop_size[1] - 1),
                                     np.logical_or(trajs[i, :, 1] < 0, trajs[i, :, 1] > self.crop_size[0] - 1))
            visibles[i, oob_inds] = 0

        return rgbs, masks, trajs, visibles, inbound

    def zigzag_just_crop(self, rgbs, masks, trajs):
        '''
        Input:
            rgbs --- list of len S, each = np.array (H, W, 3)
            trajs --- np.array (N, S, 2)
        Output:
            rgbs_aug --- np.array (S_out, H, W, 3)
            trajs_aug --- np.array (N_new, S_out, 2)
            visibles_aug --- np.array (N_new, S_out)
        '''

        N, T, _ = trajs.shape
        visibles = np.ones((N, T))

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]

        ############ create a zig zag sequence ############=
        rgbs_zigzag = []
        masks_zigzag = []
        trajs_zigzag = []
        cur_index = 0
        direction = 1

        while len(rgbs_zigzag) < self.S_out:
            # steps to go
            num_steps = np.random.randint(low=self.zigzag_step_min, high=self.zigzag_step_max + 1)
            num_steps = min(num_steps, self.S_out - len(rgbs_zigzag))

            # whether to change direction
            if np.random.rand() < self.switch_dir_prob:
                direction *= -1

            # we start at the current location (inclusive) and add frames sequentially
            for _ in range(num_steps):
                rgbs_zigzag.append(rgbs[cur_index])
                masks_zigzag.append(masks[cur_index])
                trajs_zigzag.append(trajs[:, cur_index])

                cur_index += direction

                # if this got too small (means we were at 0, and now at -1), let's bounce back the direction
                # and set cur_index to 1
                if cur_index == -1:
                    direction = 1
                    cur_index = 1

                # if this got too large (means we were at S-1, and now at S), let's bounce back the direction
                # and set cur_index to S-2
                if cur_index == S:
                    direction = -1
                    cur_index = S - 2

        rgbs = rgbs_zigzag
        masks = masks_zigzag
        trajs = np.stack(trajs_zigzag, axis=1)

        N, T, _ = trajs.shape
        visibles = np.ones((N, T))
        S = len(rgbs)
        H, W = rgbs[0].shape[:2]

        ############ spatial transform ############
        H_new = H
        W_new = W

        y0 = np.random.randint(0, H_new - self.crop_size[0])
        x0 = np.random.randint(0, W_new - self.crop_size[1])

        rgbs = [rgb[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]] for rgb in rgbs]
        masks = [mask[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]] for mask in masks]
        trajs[:, :, 0] -= x0
        trajs[:, :, 1] -= y0

        # inbound = (trajs[:,0,0] >= 0) & (trajs[:,0,0] <= self.crop_size[1]-1) & (trajs[:,0,1] >= 0) & (trajs[:,0,1] <= self.crop_size[0]-1)
        if self.force_double_inb:
            inbound0 = (trajs[:, 0, 0] >= 0) & (trajs[:, 0, 0] <= self.crop_size[1] - 1) & (trajs[:, 0, 1] >= 0) & (
                        trajs[:, 0, 1] <= self.crop_size[0] - 1)
            inbound1 = (trajs[:, 1, 0] >= 0) & (trajs[:, 1, 0] <= self.crop_size[1] - 1) & (trajs[:, 1, 1] >= 0) & (
                        trajs[:, 1, 1] <= self.crop_size[0] - 1)
            inbound = inbound0 & inbound1
        elif self.force_all_inb:
            inbound = (trajs[:, 0, 0] >= 0) & (trajs[:, 0, 0] <= self.crop_size[1] - 1) & (trajs[:, 0, 1] >= 0) & (
                        trajs[:, 0, 1] <= self.crop_size[0] - 1)
            for s in range(1, S):
                inboundi = (trajs[:, s, 0] >= 0) & (trajs[:, s, 0] <= self.crop_size[1] - 1) & (trajs[:, s, 1] >= 0) & (
                            trajs[:, s, 1] <= self.crop_size[0] - 1)
                inbound = inbound & inboundi
        else:
            inbound = (trajs[:, 0, 0] >= 0) & (trajs[:, 0, 0] <= self.crop_size[1] - 1) & (trajs[:, 0, 1] >= 0) & (
                        trajs[:, 0, 1] <= self.crop_size[0] - 1)

        trajs = trajs[inbound]
        visibles = visibles[inbound]

        # mark oob points as invisible
        for i in range(1, S):
            oob_inds = np.logical_or(np.logical_or(trajs[:, i, 0] < 0, trajs[:, i, 0] > self.crop_size[1] - 1),
                                     np.logical_or(trajs[:, i, 1] < 0, trajs[:, i, 1] > self.crop_size[0] - 1))
            visibles[oob_inds, i] = 0

        return rgbs, masks, trajs, visibles, inbound

    def __len__(self):
        # return 10
        return len(self.rgb_paths)


