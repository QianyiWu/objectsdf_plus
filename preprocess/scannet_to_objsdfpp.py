import numpy as np
import cv2
import torch
import os
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
import json
import trimesh
import glob
import PIL
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import imageio

import csv
# Code from ScanNet script to convert instance images from the *_2d-instance.zip or *_2d-instance-filt.zip data for each scan.
def read_label_mapping(filename, label_from='id', label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    # if ints convert 
    # if represents_int(mapping.keys()[0]):
    mapping = {int(k):v for k,v in mapping.items()}
    return mapping

def map_label_image(image, label_mapping):
    mapped = np.copy(image)
    for k,v in label_mapping.items():
        mapped[image==k] = v
    # merge some label like bg, wall, ceiling, floor
    # bg: 0, wall: 1, floor: 2, ceiling: 22, door: 8
    mapped[mapped==1] = 0 
    mapped[mapped==2] = 0
    mapped[mapped==22] = 0
    # add windows
    mapped[mapped==9] = 0
    # add door
    mapped[mapped==8] = 0
    # add mirror; curtain to 0
    mapped[mapped==19] = 0 # mirror
    mapped[mapped==16] = 0  # curtain
    return mapped.astype(np.uint8)

def make_instance_image(label_image, instance_image):
    output = np.zeros_like(instance_image, dtype=np.uint16)
    # oldinst2labelinst = {}
    label_instance_counts = {}
    old_insts = np.unique(instance_image)
    for inst in old_insts:
        label = label_image[instance_image==inst][0]
        if label in label_instance_counts and label !=0:
            inst_count = label_instance_counts[label] + 1
            label_instance_counts[label] = inst_count     
        else:
            inst_count = 1
            label_instance_counts[label] = inst_count
        # oldinst2labelinst[inst] = (label, inst_count)
        output[instance_image==inst] = label * 1000 + inst_count
    return output

image_size = 384
trans_totensor = transforms.Compose([
    transforms.CenterCrop(image_size*2),
    transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
])
depth_trans_totensor = transforms.Compose([
    transforms.Resize([968, 1296], interpolation=PIL.Image.NEAREST),
    transforms.CenterCrop(image_size*2),
    transforms.Resize(image_size, interpolation=PIL.Image.NEAREST),
])

seg_trans_totensor = transforms.Compose([
    transforms.CenterCrop(image_size*2),
    transforms.Resize(image_size, interpolation=PIL.Image.NEAREST),
])



out_path_prefix = '../data/custom'
data_root = '/media/hdd/monodata_ours/scans'
scenes = ['scene0050_00', 'scene0084_00', 'scene0580_00', 'scene0616_00']
# scenes = ['scene0050_00']
# out_names = ['scan1']
out_names = ['scan1', 'scan2', 'scan3', 'scan4']

label_map = read_label_mapping('/media/hdd/scannet/tasks/scannetv2-labels.combined.tsv') # path to scannet_labels.csv

for scene, out_name in zip(scenes, out_names):
    out_path = os.path.join(out_path_prefix, out_name)
    os.makedirs(out_path, exist_ok=True)
    print(out_path)

    folders = ["image", "mask", "depth", "segs"]
    for folder in folders:
        out_folder = os.path.join(out_path, folder)
        os.makedirs(out_folder, exist_ok=True)

    # load color 
    color_path = os.path.join(data_root, scene, 'color')
    color_paths = sorted(glob.glob(os.path.join(color_path, '*.jpg')), 
        key=lambda x: int(os.path.basename(x)[:-4]))
    print(color_paths)
    
    # load depth
    depth_path = os.path.join(data_root, scene, 'depth')
    depth_paths = sorted(glob.glob(os.path.join(depth_path, '*.png')),
        key=lambda x: int(os.path.basename(x)[:-4]))
    print(depth_paths)

    segs_path = os.path.join(data_root, scene, 'segs', 'instance-filt')
    segs_paths = sorted(glob.glob(os.path.join(segs_path, '*.png')),
        key=lambda x: int(os.path.basename(x)[:-4]))
    print(segs_paths)

    labels_path = os.path.join(data_root, scene, 'segs', 'label-filt')
    labels_paths = sorted(glob.glob(os.path.join(labels_path, '*.png')),
        key = lambda x: int(os.path.basename(x)[:-4]))
    print(labels_paths)

    # load intrinsic
    intrinsic_path = os.path.join(data_root, scene, 'intrinsic', 'intrinsic_color.txt')
    camera_intrinsic = np.loadtxt(intrinsic_path)
    print(camera_intrinsic)

    # load pose
    pose_path = os.path.join(data_root, scene, 'pose')
    poses = []
    pose_paths = sorted(glob.glob(os.path.join(pose_path, '*.txt')),
                        key=lambda x: int(os.path.basename(x)[:-4]))
    for pose_path in pose_paths:
        c2w = np.loadtxt(pose_path)
        poses.append(c2w)
    poses = np.array(poses)

    # deal with invalid poses
    valid_poses = np.isfinite(poses).all(axis=2).all(axis=1)
    min_vertices = poses[:, :3, 3][valid_poses].min(axis=0)
    max_vertices = poses[:, :3, 3][valid_poses].max(axis=0)
 
    center = (min_vertices + max_vertices) / 2.
    scale = 2. / (np.max(max_vertices - min_vertices) + 3.)
    print(center, scale)

    # we should normalized to unit cube
    scale_mat = np.eye(4).astype(np.float32)
    scale_mat[:3, 3] = -center
    scale_mat[:3 ] *= scale 
    scale_mat = np.linalg.inv(scale_mat)

    # copy image
    out_index = 0
    cameras = {}
    pcds = []
    H, W = 968, 1296

    # center crop by 2 * image_size
    offset_x = (W - image_size * 2) * 0.5
    offset_y = (H - image_size * 2) * 0.5
    camera_intrinsic[0, 2] -= offset_x
    camera_intrinsic[1, 2] -= offset_y
    # resize from 384*2 to 384
    resize_factor = 0.5
    camera_intrinsic[:2, :] *= resize_factor
    
    K = camera_intrinsic
    print(K)
    
    instance_semantic_mapping = {}
    instance_label_mapping = {}
    label_instance_counts = {}

    for idx, (valid, pose, depth_path, image_path, seg_path, label_path) in enumerate(zip(valid_poses, poses, depth_paths, color_paths, segs_paths, labels_paths)):
        print(idx, valid)
        if idx % 10 != 0: continue
        if not valid : continue
        
        target_image = os.path.join(out_path, "image/%06d.png"%(out_index))
        print(target_image)
        img = Image.open(image_path)
        img_tensor = trans_totensor(img)
        img_tensor.save(target_image)

        mask = (np.ones((image_size, image_size, 3)) * 255.).astype(np.uint8)

        target_image = os.path.join(out_path, "mask/%06d_mask.png"%(out_index))
        cv2.imwrite(target_image, mask)

        # load depth
        target_image = os.path.join(out_path, "depth/%06d_depth.png"%(out_index))
        depth = cv2.imread(depth_path, -1).astype(np.float32) / 1000.
        #import pdb; pdb.set_trace()
        depth_PIL = Image.fromarray(depth)
        new_depth = depth_trans_totensor(depth_PIL)
        new_depth = np.asarray(new_depth)
        plt.imsave(target_image, new_depth, cmap='viridis')
        np.save(target_image.replace(".png", ".npy"), new_depth)

        # segs 
        target_image = os.path.join(out_path, "segs/%06d_segs.png"%(out_index))
        print(target_image)
        seg = Image.open(seg_path)
        # seg_tensor = trans_totensor(seg)
        seg_tensor = seg_trans_totensor(seg)
        seg_tensor.save(target_image)
        # np.save(target_image)

        # label_mapping 
        # label = Image.open(label_path)
        # label_tensor = trans_totensor(label)
        label_np = imageio.imread(label_path)
        segs_np = imageio.imread(seg_path)
        # import pdb; pdb.set_trace()
        mapped_labels = map_label_image(label_np, label_map)
        old_insts = np.unique(segs_np)
        for inst in old_insts:
            label = mapped_labels[segs_np==inst][0]
            # import pdb; pdb.set_trace()
            instance_semantic_mapping[inst] = label
            if label == 0:
                instance_label_mapping[inst] = 0
            elif label in label_instance_counts:
                if inst not in instance_label_mapping:
                    inst_count = label_instance_counts[label] + 1 # add the number of counting for one label
                    label_instance_counts[label] = inst_count   
                    # change the instance label mapping index of this instance to 1000*label + counted number
                    instance_label_mapping[inst] = label * 1000 + inst_count 
                else: 
                    continue
            else:
                inst_count = 1
                label_instance_counts[label] = inst_count # this label never exist before, so add the inst_count as 1 and put it in label_instance_conuts
                instance_label_mapping[inst] = label * 1000 + inst_count
            


        
        # save pose
        pcds.append(pose[:3, 3])
        pose = K @ np.linalg.inv(pose)
        
        #cameras["scale_mat_%d"%(out_index)] = np.eye(4).astype(np.float32)
        cameras["scale_mat_%d"%(out_index)] = scale_mat
        cameras["world_mat_%d"%(out_index)] = pose

        out_index += 1

    # save the instance mapping file to output path 
    print({k: v for k, v in sorted(instance_label_mapping.items())})
    with open(os.path.join(out_path, 'instance_mapping.txt'), 'w') as f:
        # f.write(str(sorted(label_set)).strip('[').strip(']'))
        for k, v in instance_label_mapping.items():
            f.write(str(k)+','+str(instance_semantic_mapping[k])+','+str(v)+'\n')
    f.close()

    #np.savez(os.path.join(out_path, "cameras_sphere.npz"), **cameras)
    np.savez(os.path.join(out_path, "cameras.npz"), **cameras)
