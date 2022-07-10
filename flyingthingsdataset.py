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
    def __init__(self, dataset_location='../../aharley/flyingthings', dset='TRAIN', subset='all', use_augs=False, N=0, S=4, crop_size=(368, 496),
                 version='ab', occ_version='ad', force_inb=True, force_double_inb=False, force_all_inb=False, N_min=None):

        print('loading FlyingThingsDataset...')
        
        self.S = S
        self.N = N
        if N_min is None:
            self.N_min = self.N//4
        else:
            self.N_min = N_min
            
        self.use_augs = use_augs
        self.load_flow = False
        
        self.rgb_paths = []
        self.traj_paths = []
        self.mask_paths = []
        self.flow_f_paths = []
        self.flow_b_paths = []
        self.start_inds = []
        self.load_fails = []

        self.force_inb = force_inb
        self.force_double_inb = force_double_inb
        self.force_all_inb = force_all_inb

        if subset=='all':
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
                                
                                # print('file_size', file_size)
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
                                #     if self.load_flow:
                                #         self.flow_f_paths.append(cur_flow_f_path)
                                #         self.flow_b_paths.append(cur_flow_b_path)
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
                                if self.load_flow:
                                    self.flow_f_paths.append(cur_flow_f_path)
                                    self.flow_b_paths.append(cur_flow_b_path)
                                sys.stdout.write('.')
                                sys.stdout.flush()
                            
                if ii > 10 and (ii % 100)==0:
                    # print('found %d samples in %s so far...' % (len(self.rgb_paths), dataset_location))
                    sys.stdout.write('%.1f%%' % (100*(ii+1)/len(folder_names)))
                    sys.stdout.flush()
        # print('%.1f%%' % (100*(ii+1)/len(folder_names)))
        print('found %d samples in %s' % (len(self.rgb_paths), dataset_location))


        # we also need to step through and collect ooccluder info
        print('loading occluders...')

        self.occ_rgb_paths = []
        self.occ_mask_paths = []
        self.occ_start_inds = []
        self.occ_traj_paths = []


        print('locking start_ind=0, for speed')
        for subset in subsets:
            occ_root_path = os.path.join(dataset_location, "occluders_%s" % occ_version, dset, subset)

            folder_names = [folder.split('/')[-1] for folder in glob.glob(os.path.join(occ_root_path, "*"))]
            folder_names = sorted(folder_names)
            # print('folder_names', folder_names)
            
            for ii, folder_name in enumerate(folder_names):
                
                for lr in ['left', 'right']:

                    cur_rgb_path = os.path.join(rgb_root_path, folder_name, lr)
                    cur_mask_path = os.path.join(mask_root_path, folder_name, lr)
                    
                    cur_occ_path = os.path.join(occ_root_path, folder_name, lr)
                    for start_ind in [0,1,2]:
                    # start_ind = 0
                    # if True:
                        
                        file_names = glob.glob(os.path.join(cur_occ_path, "*at_%d.npz" % start_ind))
                        # print('file_names', file_names)
                        # print('len', len(file_names))
                        # input()
                                               
                        for ii in range(len(file_names)): 
                            occ_fn = cur_occ_path + '/occluder_%d_at_%d.npz' % (ii, start_ind)
                            if os.path.isfile(occ_fn):
                                file_size = os.path.getsize(occ_fn)
                                if file_size > 100: # the empty ones are 10 bytes
                                    self.occ_traj_paths.append(occ_fn)
                                    self.occ_rgb_paths.append(cur_rgb_path)
                                    self.occ_mask_paths.append(cur_mask_path)
                                    self.occ_start_inds.append(start_ind)

                sys.stdout.write('.')
                sys.stdout.flush()
                            
                if ii > 10 and (ii % 100)==0:
                    # print('found %d samples in %s so far...' % (len(self.rgb_paths), dataset_location))
                    sys.stdout.write('%.1f%%' % (100*(ii+1)/len(folder_names)))
                    sys.stdout.flush()
        # print('%.1f%%' % (100*(ii+1)/len(folder_names)))
        print('found %d occluders in %s' % (len(self.occ_rgb_paths), dataset_location))
        

        # photometric augmentation
        # self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.photo_aug = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.25/3.14)
        self.blur_aug = GaussianBlur(11, sigma=(0.1, 2.0))
        
        self.blur_prob = 0.2
        self.asymmetric_color_aug_prob = 0.2

        # occlusion augmentation
        self.eraser_aug_prob = 0.5
        self.eraser_bounds = [20, 300]

        # spatial augmentations
        self.crop_size = crop_size
        self.min_scale = -0.1 # 2^this
        self.max_scale = 1.0 # 2^this
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2
        self.do_flip = True
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1
        self.max_crop_offset = 3
        
    def __getitem__(self, index):
        gotit = False
        
        updated_fails = False
            
        while not gotit:
            # if this index is a known failure, re-index
            load_fail = self.load_fails[index]
            while load_fail:
                index = np.random.randint(0, len(self.load_fails))
                load_fail = self.load_fails[index]
            
            cur_rgb_path = self.rgb_paths[index]
            cur_traj_path = self.traj_paths[index]
            cur_mask_path = self.mask_paths[index]
            if self.load_flow:
                cur_flow_f_path = self.flow_f_paths[index]
                cur_flow_b_path = self.flow_b_paths[index]
            # print('cur_rgb_path', cur_rgb_path)

            start_ind = self.start_inds[index]

            img_names = [folder.split('/')[-1].split('.')[0] for folder in glob.glob(os.path.join(cur_rgb_path, "*"))]
            img_names = sorted(img_names)
            img_names = img_names[start_ind:start_ind+self.S]

            # start_ind = np.random.randint(len(img_names)-self.S)

            trajs = np.load(os.path.join(cur_traj_path, 'trajs_at_%d.npz' % start_ind), allow_pickle=True)
            trajs = dict(trajs)['trajs'] # S, N 2
            trajs = trajs.astype(np.float32)
            S, N, D = trajs.shape
            assert(S==self.S)
            valids = np.ones((self.S, N)).astype(np.float32)

            if N > self.N*2:

                rgbs = []
                masks = []
                flows_f = []
                flows_b = []

                for img_name in img_names:
                    # im = Image.open(os.path.join(cur_rgb_path, '{0}.webp'.format(img_name))) # H, W, 3
                    with Image.open(os.path.join(cur_rgb_path, '{0}.webp'.format(img_name))) as im:
                        rgbs.append(np.array(im))

                    # print('rgb', rgbs[-1].shape)
                    # print('cur_mask_path', cur_mask_path)
                    mask = readImage(os.path.join(cur_mask_path, '{0}.pfm'.format(img_name)))
                    # print('%s mask' % img_name, mask.shape)
                    masks.append(mask)

                    if self.load_flow:
                        # print('cur_flow_f_path', cur_flow_f_path)
                        if 'left' in cur_flow_f_path:
                            flows_f.append(readPFM(os.path.join(cur_flow_f_path, 'OpticalFlowIntoFuture_{0}_L.pfm'.format(img_name)))[:, :, :2])
                        else:
                            flows_f.append(readPFM(os.path.join(cur_flow_f_path, 'OpticalFlowIntoFuture_{0}_R.pfm'.format(img_name)))[:, :, :2])
                if self.load_flow:
                    flows = flows_f
                else:
                    flows = None

                # N_, S_, D_ = trajs.shape
                # if N_ < self.N:
                #     print('N_', N_)
                # assert(N_ >= self.N)
                # assert(S_ >= self.S)

                # trajs = trajs[:, :self.S]
                # print('trajs', trajs.shape)

                # print('rgbs[0]', rgbs[0].shape)
                # print('flows[0]', flows[0].shape)
                # print('masks[0]', masks[0].shape)

                if self.use_augs:
                    success = False
                    for _ in range(5):
                        rgbs_aug, masks_aug, trajs_aug, visibles_aug, valids_aug, _ = self.augment(rgbs, masks, trajs, valids)
                        try:
                            print('trajs_aug', trajs_aug.shape)
                            N_ = trajs_aug.shape[0]
                            if N_ >= self.N:
                                success = True
                            break
                        except:
                            # print('retrying aug')
                            continue
                    rgbs, masks, trajs, visibles, valids = rgbs_aug, masks_aug, trajs_aug, visibles_aug, valids_aug
                else:
                    for _ in range(5):
                        rgbs_crop, masks_crop, trajs_crop, visibles_crop, valids_crop, _ = self.just_crop(rgbs, masks, trajs, valids)
                        try:
                            N_ = trajs_crop.shape[0]
                            if N_ >= self.N:
                                success = True
                            break
                        except:
                            continue
                    rgbs, masks, trajs, visibles, valids = rgbs_crop, masks_crop, trajs_crop, visibles_crop, valids_crop

                # print('trajs', trajs.shape)
                # print('self.N', self.N)
                N_ = min(trajs.shape[1], self.N)

                if N_ > self.N_min:
                    # print('N_', N_)
                    # traj_id = np.random.choice(trajs.shape[1], size=N_, replace=False)
                    traj_id = utils.py.farthest_point_sample(trajs[0], N_, deterministic=False)

                    # trajs = trajs[traj_id] # N, S, 2

                    trajs_full = np.zeros((self.S, self.N, 2)).astype(np.float32)
                    valids_full = np.zeros((self.S, self.N)).astype(np.float32)
                    visibles_full = np.zeros((self.S, self.N)).astype(np.float32)
                    # valids = np.zeros((self.N)).astype(np.float32)
                    trajs_full[:,:N_] = trajs[:,traj_id]
                    valids_full[:,:N_] = valids[:,traj_id]
                    visibles_full[:,:N_] = visibles[:,traj_id]

                    rgbs = torch.from_numpy(np.stack(rgbs, 0)).permute(0, 3, 1, 2) # S, C, H, W
                    # if self.load_flow:
                    #     flows = torch.from_numpy(np.stack(flows, 0)).permute(0, 3, 1, 2) # S, C, H, W
                    masks = torch.from_numpy(np.stack(masks, 0)).unsqueeze(1) # S, 1, H, W
                    trajs = torch.from_numpy(trajs_full) # S, N, 2
                    valids = torch.from_numpy(valids_full) # S, N
                    visibles = torch.from_numpy(visibles_full)

                    if torch.sum(valids[0,:]) > self.N_min:
                        gotit = True

            if not gotit:
                # save time by not trying this index again
                load_fail = 1
                self.load_fails[index] = load_fail
                # print('updated load_fails (on this worker): %d/%d...' % (np.sum(self.load_fails), len(self.load_fails)))
                
                updated_fails = True
        # end while not gotit
                
        # print('torch.sum(valids)', torch.sum(valids))
        return_dict = {
            # 'cur_heavy_path': cur_heavy_path, 
            'rgbs': rgbs,
            'masks': masks,
            'trajs': trajs,
            # 'trajs_XYs': [trajs_XYs],
            # 'trajs_Ts': [trajs_Ts],
            'valids': valids,
            'visibles': visibles,
            'updated_fails': updated_fails,
        }
        # if self.load_flow:
        #     return_dict['flows'] = flows
        return return_dict
        

    def augment(self, rgbs, masks, trajs, valids):
        '''
        Input:
            rgbs --- list of len S, each = np.array (H, W, 3)
            trajs --- np.array (S,N,2)
        Output:
            rgbs_aug --- np.array (S_out, H, W, 3)
            trajs_aug --- np.array (S, N_new, 2)
            visibles_aug --- np.array (S, N_new)
        '''
        T, N, _ = trajs.shape
        visibles = np.ones((T, N))

        # print('trajs', trajs.shape)
        # print('len(rgbs)', len(rgbs))
        
        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert(S==T)

        rgbs = [0.1*rgb.astype(np.float32) for rgb in rgbs]
        # rgbs = [rgb.astype(np.float32) for rgb in rgbs]
        
        ############ occluders from other videos ############
        # for _ in range(np.random.randint(1, 3)): # number of occluders:
        for _ in range(5): # number of occluders:
            # i want to get an alt rgb, and alt mask
            alt_ind = np.random.choice(len(self.occ_rgb_paths))
            occ_traj_path = self.occ_traj_paths[alt_ind]
            occ_rgb_path = self.occ_rgb_paths[alt_ind]
            occ_mask_path = self.occ_mask_paths[alt_ind]
            occ_start_ind = self.occ_start_inds[alt_ind]
            
            img_names = [folder.split('/')[-1].split('.')[0] for folder in glob.glob(os.path.join(occ_rgb_path, "*"))]
            img_names = sorted(img_names)
            img_names = img_names[occ_start_ind:occ_start_ind+self.S]

            occ_info = np.load(occ_traj_path, allow_pickle=True)
            occ_info = dict(occ_info)
            alt_trajs = occ_info['trajs'] # S,N,2, or None
            occ_id = occ_info['id_'] # []

            if alt_trajs is not None:
                # import ipdb; ipdb.set_trace()
                # print('alt_trajs', alt_trajs.shape)
                if (np.any(alt_trajs) is not None) and alt_trajs.shape[1] > 0:
                    alt_visibles = np.ones_like(alt_trajs[:,:,0]) # S,N
                    alt_valids = np.ones_like(alt_trajs[:,:,0]) # S,N

            alt_rgbs = []
            alt_masks = []
            for img_name in img_names:
                with Image.open(os.path.join(occ_rgb_path, '{0}.webp'.format(img_name))) as im:
                    alt_rgbs.append(np.array(im))
                mask = readImage(os.path.join(occ_mask_path, '{0}.pfm'.format(img_name)))
                mask = (mask==occ_id).astype(np.float32).reshape(H, W, 1)
                # print('mask', mask.shape)
                alt_masks.append(mask)

            # alt_rgbs = [np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in alt_rgbs]
            # if np.random.rand() < self.blur_prob:
            #     alt_rgbs = [np.array(self.blur_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in alt_rgbs]
            
                # xy0 = alt_trajs[:,0].round().astype(np.int32) # N, 2
                # x0, y0 = xy0[:,0], xy0[:,1]
                # inds = alt_masks[0][y0,x0,0] > 0 # use the TRUE mask here
                # alt_trajs = alt_trajs[inds]
                # N_ = alt_trajs.shape[0]
                # alt_visibles = np.ones((N_, T))
                # alt_valids = np.ones((N_, T))

                # if self.version2 is not None:
                #     cur_traj2_path = self.traj2_paths[index]
                #     alt_traj2s = np.load(os.path.join(cur_traj2_path, 'traj.npz'), allow_pickle=True)
                #     alt_traj2s = dict(alt_traj2s)['trajs'] # N2, S, 2

                #     xy0 = alt_traj2s[:,0].round().astype(np.int32) # N, 2
                #     x0, y0 = xy0[:,0], xy0[:,1]
                #     inds = alt_masks[0][y0,x0,0] > 0
                #     alt_traj2s = alt_traj2s[inds]
                #     N2_ = alt_traj2s.shape[0]
                #     S2_ = alt_traj2s.shape[1]

                #     full_trajs = np.zeros((N_+N2_, self.S, 2)).astype(np.float32)
                #     full_valids = np.zeros((N_+N2_, self.S)).astype(np.float32)

                #     full_trajs[:N_] = alt_trajs
                #     full_valids[:N_] = alt_valids
                #     full_visibles[:N_] = alt_visibles

                #     full_trajs[N_:,:S2] = alt_traj2s
                #     full_valids[N_:,:S2] = alt_valid2s
                #     full_visibles[N_:,:S2] = traj2s
                #     valids[:N1,:] = 1
                #     valids[N1:,:S2] = 1


            # print('mask', alt_masks[0].shape)
            # rgbs = [mask*rgb for (mask,rgb) in zip(alt_masks,alt_rgbs)]
            rgbs = [rgb*(1.0-alt_mask)+alt_rgb*alt_mask for (rgb,alt_rgb,alt_mask) in zip(rgbs,alt_rgbs,alt_masks)]

            # # darken the non-occluder, for debug
            # rgbs = [rgb*(1.0-(alt_mask*0.5)) for (rgb,alt_rgb,alt_mask) in zip(rgbs,alt_rgbs,alt_masks)]

            # any traj in the masks should be marked invisible
            for s in range(S):
                xy = trajs[s].round().astype(np.int32) # N, 2
                x, y = xy[:,0], xy[:,1]
                x = x.clip(0, W-1)
                y = y.clip(0, H-1)
                # this may generously mark it as OK, but we will catch the OOB guys later
                # use the WIDE mask here, since this is for occlusion
                inds = alt_masks[s][y,x,0] > 0
                visibles[s,inds] = 0

            rgbs = [rgb.astype(np.uint8) for rgb in rgbs]

            if alt_trajs is not None:
                if (np.any(alt_trajs) is not None) and alt_trajs.shape[1] > 0:
                    trajs = np.concatenate([trajs, alt_trajs], axis=1)
                    visibles = np.concatenate([visibles, alt_visibles], axis=1)
                    valids = np.concatenate([valids, alt_valids], axis=1)
        
        # ############ photometric augmentation ############
        # if np.random.rand() < self.asymmetric_color_aug_prob:
        #     for i in range(S):
        #         rgbs[i] = np.array(self.photo_aug(Image.fromarray(rgbs[i])), dtype=np.uint8)
        # else:
        #     rgbs = [np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]

        # if np.random.rand() < self.blur_prob:
        #     rgbs = [np.array(self.blur_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]
        
        # ############ eraser transform (per image) ############
        # for i in range(1, S):
        #     if np.random.rand() < self.eraser_aug_prob:
        #         mean_color = np.mean(rgbs[i].reshape(-1, 3), axis=0)
        #         for _ in range(np.random.randint(1, 6)):
        #             xc = np.random.randint(0, W)
        #             yc = np.random.randint(0, H)
        #             dx = np.random.randint(self.eraser_bounds[0], self.eraser_bounds[1])
        #             dy = np.random.randint(self.eraser_bounds[0], self.eraser_bounds[1])
        #             x0 = np.clip(xc - dx/2, 0, W-1).round().astype(np.int32)
        #             x1 = np.clip(xc + dx/2, 0, W-1).round().astype(np.int32)
        #             y0 = np.clip(yc - dy/2, 0, W-1).round().astype(np.int32)
        #             y1 = np.clip(yc + dy/2, 0, W-1).round().astype(np.int32)
        #             # print(x0, x1, y0, y1)
        #             rgbs[i][y0:y1, x0:x1, :] = mean_color

        #             occ_inds = np.logical_and(np.logical_and(trajs[:,i,0] >= x0, trajs[:,i,0] < x1), np.logical_and(trajs[:,i,1] >= y0, trajs[:,i,1] < y1))
        #             visibles[occ_inds, i] = 0

        # ############ spatial transform ############
        # # scaling + stretching
        # scale_x = 1.0
        # scale_y = 1.0
        # H_new = H
        # W_new = W
        # if np.random.rand() < self.spatial_aug_prob:
        #     # print('spat')
        #     min_scale = np.maximum(
        #         (self.crop_size[0] + 8) / float(H),
        #         (self.crop_size[1] + 8) / float(W))

        #     scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        #     scale_x = scale 
        #     scale_y = scale 

        #     if np.random.rand() < self.stretch_prob:
        #         # print('stretch')
        #         scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        #         scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        #     scale_x = np.clip(scale_x, min_scale, None)
        #     scale_y = np.clip(scale_y, min_scale, None)

        #     H_new = int(H * scale_y)
        #     W_new = int(W * scale_x)
        #     # dim_resize = (W_new, H_new * S)
        #     rgbs = [cv2.resize(rgb, (W_new, H_new), interpolation=cv2.INTER_LINEAR) for rgb in rgbs]
        #     masks = [cv2.resize(mask, (W_new, H_new), interpolation=cv2.INTER_LINEAR) for mask in masks]
        # trajs[:,:,0] *= scale_x
        # trajs[:,:,1] *= scale_y

        # # flip
        # h_flipped = False
        # v_flipped = False
        # if self.do_flip:
        #     # h flip
        #     if np.random.rand() < self.h_flip_prob:
        #         # print('h flip')
        #         h_flipped = True
        #         rgbs = [rgb[:,::-1] for rgb in rgbs]
        #         masks = [mask[:,::-1] for mask in masks]
        #     # v flip
        #     if np.random.rand() < self.v_flip_prob:
        #         # print('v flip')
        #         v_flipped = True
        #         rgbs = [rgb[::-1] for rgb in rgbs]
        #         masks = [mask[::-1] for mask in masks]
        # if h_flipped:
        #     trajs[:,:,0] = W_new - trajs[:,:,0]
        # if v_flipped:
        #     trajs[:,:,1] = H_new - trajs[:,:,1]

        # y0 = np.random.randint(0, H_new - self.crop_size[0])
        # x0 = np.random.randint(0, W_new - self.crop_size[1])
        # for s in range(S):
        #     if s > 0:
        #         x0 = x0 + np.random.randint(-self.max_crop_offset, self.max_crop_offset+1)
        #         y0 = y0 + np.random.randint(-self.max_crop_offset, self.max_crop_offset+1)
        #     y0 = min(max(0, y0), H_new - self.crop_size[0] - 1)
        #     x0 = min(max(0, x0), W_new - self.crop_size[1] - 1)
        #     rgbs[s] = rgbs[s][y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        #     masks[s] = masks[s][y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        #     trajs[:,s,0] -= x0
        #     trajs[:,s,1] -= y0
        # # print('-')

        # if self.force_double_inb:
        #     inbound0 = (trajs[:,0,0] >= 0) & (trajs[:,0,0] <= self.crop_size[1]-1) & (trajs[:,0,1] >= 0) & (trajs[:,0,1] <= self.crop_size[0]-1)
        #     inbound1 = (trajs[:,1,0] >= 0) & (trajs[:,1,0] <= self.crop_size[1]-1) & (trajs[:,1,1] >= 0) & (trajs[:,1,1] <= self.crop_size[0]-1)
        #     inbound = inbound0 & inbound1
        # elif self.force_all_inb:
        #     inbound = (trajs[:,0,0] >= 0) & (trajs[:,0,0] <= self.crop_size[1]-1) & (trajs[:,0,1] >= 0) & (trajs[:,0,1] <= self.crop_size[0]-1)
        #     for s in range(1,S):
        #         inboundi = (trajs[:,s,0] >= 0) & (trajs[:,s,0] <= self.crop_size[1]-1) & (trajs[:,s,1] >= 0) & (trajs[:,s,1] <= self.crop_size[0]-1)
        #         inbound = inbound & inboundi
        # else:
        #     inbound = (trajs[:,0,0] >= 0) & (trajs[:,0,0] <= self.crop_size[1]-1) & (trajs[:,0,1] >= 0) & (trajs[:,0,1] <= self.crop_size[0]-1)

        # # ensure visible on first timestep
        # inbound = inbound & (visibles[:,0] > 0)

        # trajs = trajs[inbound]
        # visibles = visibles[inbound]
        # valids = valids[inbound]
        # # mark oob points as invisible
        # for i in range(1, S):
        #     oob_inds = np.logical_or(np.logical_or(trajs[:,i,0] < 0, trajs[:,i,0] > self.crop_size[1]-1), np.logical_or(trajs[:,i,1] < 0, trajs[:,i,1] > self.crop_size[0]-1))
        #     visibles[oob_inds,i] = 0



        ############ spatial transform ############
        H_new = H
        W_new = W

        if True:
            # simple crop
            y0 = np.random.randint(0, H_new - self.crop_size[0])
            x0 = np.random.randint(0, W_new - self.crop_size[1])
            rgbs = [rgb[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]] for rgb in rgbs]
            masks = [mask[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]] for mask in masks]
            trajs[:,:,0] -= x0
            trajs[:,:,1] -= y0
        else:
            # per-timestep crop
            y0 = np.random.randint(0, H_new - self.crop_size[0])
            x0 = np.random.randint(0, W_new - self.crop_size[1])
            for s in range(S):
                if s > 0:
                    x0 = x0 + np.random.randint(-self.max_crop_offset, self.max_crop_offset+1)
                    y0 = y0 + np.random.randint(-self.max_crop_offset, self.max_crop_offset+1)
                y0 = min(max(0, y0), H_new - self.crop_size[0] - 1)
                x0 = min(max(0, x0), W_new - self.crop_size[1] - 1)
                rgbs[s] = rgbs[s][y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
                masks[s] = masks[s][y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
                trajs[s,:,0] -= x0
                trajs[s,:,1] -= y0

        # inbound = (trajs[:,0,0] >= 0) & (trajs[:,0,0] <= self.crop_size[1]-1) & (trajs[:,0,1] >= 0) & (trajs[:,0,1] <= self.crop_size[0]-1)
        if self.force_double_inb:
            inbound0 = (trajs[0,:,0] >= 0) & (trajs[0,:,0] <= self.crop_size[1]-1) & (trajs[0,:,1] >= 0) & (trajs[0,:,1] <= self.crop_size[0]-1)
            inbound1 = (trajs[1,:,0] >= 0) & (trajs[1,:,0] <= self.crop_size[1]-1) & (trajs[1,:,1] >= 0) & (trajs[1,:,1] <= self.crop_size[0]-1)
            inbound = inbound0 & inbound1
        elif self.force_all_inb:
            inbound = (trajs[0,:,0] >= 0) & (trajs[0,:,0] <= self.crop_size[1]-1) & (trajs[0,:,1] >= 0) & (trajs[0,:,1] <= self.crop_size[0]-1)
            for s in range(1,S):
                inboundi = (trajs[:,s,0] >= 0) & (trajs[:,s,0] <= self.crop_size[1]-1) & (trajs[:,s,1] >= 0) & (trajs[:,s,1] <= self.crop_size[0]-1)
                inbound = inbound & inboundi
        else:
            inbound = (trajs[0,:,0] >= 0) & (trajs[0,:,0] <= self.crop_size[1]-1) & (trajs[0,:,1] >= 0) & (trajs[0,:,1] <= self.crop_size[0]-1)
        
        inbound = inbound & (visibles[0] > 0)
        
        trajs = trajs[:,inbound]
        visibles = visibles[:,inbound]
        valids = valids[:,inbound]
        
        # mark oob points as invisible
        for s in range(1, S):
            oob_inds = np.logical_or(np.logical_or(trajs[s,:,0] < 0, trajs[s,:,0] > self.crop_size[1]-1), np.logical_or(trajs[s,:,1] < 0, trajs[s,:,1] > self.crop_size[0]-1))
            visibles[s,oob_inds] = 0

        return rgbs, masks, trajs, visibles, valids, inbound
    
    def just_crop(self, rgbs, masks, trajs, valids):
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
        visibles = np.ones((T, N))
        
        S = len(rgbs)
        H, W = rgbs[0].shape[:2]

        ############ spatial transform ############
        H_new = H
        W_new = W

        if True:
            # simple crop
            y0 = np.random.randint(0, H_new - self.crop_size[0])
            x0 = np.random.randint(0, W_new - self.crop_size[1])
            rgbs = [rgb[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]] for rgb in rgbs]
            masks = [mask[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]] for mask in masks]
            trajs[:,:,0] -= x0
            trajs[:,:,1] -= y0
        else:
            # per-timestep crop
            y0 = np.random.randint(0, H_new - self.crop_size[0])
            x0 = np.random.randint(0, W_new - self.crop_size[1])
            for s in range(S):
                if s > 0:
                    x0 = x0 + np.random.randint(-self.max_crop_offset, self.max_crop_offset+1)
                    y0 = y0 + np.random.randint(-self.max_crop_offset, self.max_crop_offset+1)
                y0 = min(max(0, y0), H_new - self.crop_size[0] - 1)
                x0 = min(max(0, x0), W_new - self.crop_size[1] - 1)
                rgbs[s] = rgbs[s][y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
                masks[s] = masks[s][y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
                trajs[s,:,0] -= x0
                trajs[s,:,1] -= y0

        # inbound = (trajs[:,0,0] >= 0) & (trajs[:,0,0] <= self.crop_size[1]-1) & (trajs[:,0,1] >= 0) & (trajs[:,0,1] <= self.crop_size[0]-1)
        if self.force_double_inb:
            inbound0 = (trajs[0,:,0] >= 0) & (trajs[0,:,0] <= self.crop_size[1]-1) & (trajs[0,:,1] >= 0) & (trajs[0,:,1] <= self.crop_size[0]-1)
            inbound1 = (trajs[1,:,0] >= 0) & (trajs[1,:,0] <= self.crop_size[1]-1) & (trajs[1,:,1] >= 0) & (trajs[1,:,1] <= self.crop_size[0]-1)
            inbound = inbound0 & inbound1
        elif self.force_all_inb:
            inbound = (trajs[0,:,0] >= 0) & (trajs[0,:,0] <= self.crop_size[1]-1) & (trajs[0,:,1] >= 0) & (trajs[0,:,1] <= self.crop_size[0]-1)
            for s in range(1,S):
                inboundi = (trajs[:,s,0] >= 0) & (trajs[:,s,0] <= self.crop_size[1]-1) & (trajs[:,s,1] >= 0) & (trajs[:,s,1] <= self.crop_size[0]-1)
                inbound = inbound & inboundi
        else:
            inbound = (trajs[0,:,0] >= 0) & (trajs[0,:,0] <= self.crop_size[1]-1) & (trajs[0,:,1] >= 0) & (trajs[0,:,1] <= self.crop_size[0]-1)
        
        trajs = trajs[:,inbound]
        visibles = visibles[:,inbound]
        valids = valids[:,inbound]
        
        # mark oob points as invisible
        for s in range(1, S):
            oob_inds = np.logical_or(np.logical_or(trajs[s,:,0] < 0, trajs[s,:,0] > self.crop_size[1]-1), np.logical_or(trajs[s,:,1] < 0, trajs[s,:,1] > self.crop_size[0]-1))
            visibles[s,oob_inds] = 0

        return rgbs, masks, trajs, visibles, valids, inbound

    def __len__(self):
        # return 10
        return len(self.rgb_paths)


