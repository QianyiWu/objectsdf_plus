import os
import torch
import torch.nn.functional as F
import numpy as np

import utils.general as utils
from utils import rend_util
from glob import glob
import cv2
import random

# Dataset with monocular depth, normal and segmentation mask
class SceneDatasetDN_segs(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 img_res,
                 scan_id=0,
                 center_crop_type='xxxx',
                 use_mask=False,
                 num_views=-1
                 ):

        self.instance_dir = os.path.join('../data', data_dir, 'scan{0}'.format(scan_id))
        print(self.instance_dir)

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res
        self.num_views = num_views
        assert num_views in [-1, 3, 6, 9]
        
        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None
        
        def glob_data(data_dir):
            data_paths = []
            data_paths.extend(glob(data_dir))
            data_paths = sorted(data_paths)
            return data_paths
            
        image_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_rgb.png"))
        depth_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_depth.npy"))
        normal_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_normal.npy"))
        # semantic_paths
        semantic_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "segs", "*_segs.png"))
        
        # mask is only used in the replica dataset as some monocular depth predictions have very large error and we ignore it
        if use_mask:
            mask_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_mask.npy"))
        else:
            mask_paths = None

        self.n_images = len(image_paths)
        print('[INFO]: Dataset Size ', self.n_images)

        # load instance_mapping_dict
        self.label_mapping = None
        self.instance_mapping_dict= {}
        # with open(os.path.join(data_dir, 'label_mapping_instance.txt'), 'r') as f:
        #     content = f.readlines()
        #     self.label_mapping = [int(a) for a in content[0].split(',')]

        # using the remapped instance label for training
        with open(os.path.join(self.instance_dir, 'instance_mapping.txt'), 'r') as f:
            for l in f:
                (k, v_sem, v_ins) = l.split(',')
                self.instance_mapping_dict[int(k)] = int(v_ins)
        # self.label_mapping = [] # get the sorted label mapping list
        # for k, v in self.instance_mapping_dict.items():
        #     if v not in self.label_mapping: # not a duplicate instance
        #         self.label_mapping.append(v)
        self.label_mapping = sorted(set(self.instance_mapping_dict.values())) # get sorted label mapping. The first one is the background
        # sorted
        # print('Instance Label Mapping: ', self.label_mapping)
        self.obj_id = torch.from_numpy(np.array(range(len(self.label_mapping))))
        print(self.obj_id)
        
        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)

            # because we do resize and center crop 384x384 when using omnidata model, we need to adjust the camera intrinsic accordingly
            if center_crop_type == 'center_crop_for_replica':
                scale = 384 / 680
                offset = (1200 - 680 ) * 0.5
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'center_crop_for_tnt':
                scale = 384 / 540
                offset = (960 - 540) * 0.5
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'center_crop_for_dtu':
                scale = 384 / 1200
                offset = (1600 - 1200) * 0.5
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'padded_for_dtu':
                scale = 384 / 1200
                offset = 0
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'no_crop':  # for scannet dataset, we already adjust the camera intrinsic duing preprocessing so nothing to be done here
                pass
            else:
                raise NotImplementedError
            
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.rgb_images = []
        for path in image_paths:
            rgb = rend_util.load_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())
            
        self.depth_images = []
        self.normal_images = []

        for dpath, npath in zip(depth_paths, normal_paths):
            depth = np.load(dpath)
            self.depth_images.append(torch.from_numpy(depth.reshape(-1, 1)).float())
        
            normal = np.load(npath)
            normal = normal.reshape(3, -1).transpose(1, 0)
            # important as the output of omnidata is normalized
            normal = normal * 2. - 1.
            self.normal_images.append(torch.from_numpy(normal).float())

        # load semantic
        self.semantic_images = []
        for spath in semantic_paths:
            semantic_ori = cv2.imread(spath, cv2.IMREAD_UNCHANGED).astype(np.int32)
            semantic = np.copy(semantic_ori)
            ins_list = np.unique(semantic_ori)
            if self.label_mapping is not None:
                for i in ins_list:
                    semantic[semantic_ori ==i] = self.label_mapping.index(self.instance_mapping_dict[i])
            self.semantic_images.append(torch.from_numpy(semantic.reshape(-1, 1)).float())

        # load mask
        self.mask_images = []
        if mask_paths is None:
            for depth in self.depth_images:
                mask = torch.ones_like(depth)
                self.mask_images.append(mask)
        else:
            for path in mask_paths:
                mask = np.load(path)
                self.mask_images.append(torch.from_numpy(mask.reshape(-1, 1)).float())

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        if self.num_views >= 0:
            image_ids = [25, 22, 28, 40, 44, 48, 0, 8, 13][:self.num_views]
            idx = image_ids[random.randint(0, self.num_views - 1)]
        
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx]
        }
        
        ground_truth = {
            "rgb": self.rgb_images[idx],
            "depth": self.depth_images[idx],
            "mask": self.mask_images[idx],
            "normal": self.normal_images[idx],
            "segs": self.semantic_images[idx]
        }

        if self.sampling_idx is not None:
            if (self.random_image_for_path is None) or (idx not in self.random_image_for_path):
                # print('sampling_idx:', self.sampling_idx)
                ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
                ground_truth["full_rgb"] = self.rgb_images[idx]
                ground_truth["normal"] = self.normal_images[idx][self.sampling_idx, :]
                ground_truth["depth"] = self.depth_images[idx][self.sampling_idx, :]
                ground_truth["full_depth"] = self.depth_images[idx]
                ground_truth["mask"] = self.mask_images[idx][self.sampling_idx, :]
                ground_truth["full_mask"] = self.mask_images[idx]
                ground_truth["segs"] = self.semantic_images[idx][self.sampling_idx, :]
            
                sample["uv"] = uv[self.sampling_idx, :]
                sample["is_patch"] = torch.tensor([False])
            else:
                # sampling a patch from the image, this could be used for training with depth total variational loss
                # a fix patch sampling, which require the sampling_size should be a H*H continuous path
                patch_size = np.floor(np.sqrt(len(self.sampling_idx))).astype(np.int32)
                start = np.random.randint(self.img_res[1]-patch_size +1)*self.img_res[0] + np.random.randint(self.img_res[1]-patch_size +1) # the start coordinate
                idx_row = torch.arange(start, start + patch_size)
                patch_sampling_idx = torch.cat([idx_row + self.img_res[1]*m for m in range(patch_size)])
                ground_truth["rgb"] = self.rgb_images[idx][patch_sampling_idx, :]
                ground_truth["full_rgb"] = self.rgb_images[idx]
                ground_truth["normal"] = self.normal_images[idx][patch_sampling_idx, :]
                ground_truth["depth"] = self.depth_images[idx][patch_sampling_idx, :]
                ground_truth["full_depth"] = self.depth_images[idx]
                ground_truth["mask"] = self.mask_images[idx][patch_sampling_idx, :]
                ground_truth["full_mask"] = self.mask_images[idx]
                ground_truth["segs"] = self.semantic_images[idx][patch_sampling_idx, :]
            
                sample["uv"] = uv[patch_sampling_idx, :]
                sample["is_patch"] = torch.tensor([True])
        
        return idx, sample, ground_truth


    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size, sampling_pattern='random'):
        if sampling_size == -1:
            self.sampling_idx = None
            self.random_image_for_path = None
        else:
            if sampling_pattern == 'random':
                self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]
                self.random_image_for_path = None
            elif sampling_pattern == 'patch':
                self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]
                self.random_image_for_path = torch.randperm(self.n_images, )[:int(self.n_images/10)]
            else:
                raise NotImplementedError('the sampling pattern is not implemented.')

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']