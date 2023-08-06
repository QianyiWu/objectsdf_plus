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

# For Replica dataset, we adopt the camera intrinsic/extrinsic/rgb/depth/normal from MonoSDF dataset
# For instance label, we use the instance label from vMAP processed dataset.

# map the instance segmentation result to semantic segmentation result

image_size = 384
# trans_totensor = transforms.Compose([
#     transforms.CenterCrop(image_size),
#     transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
# ])
# depth_trans_totensor = transforms.Compose([
#     # transforms.Resize([680, 1200], interpolation=PIL.Image.NEAREST),
#     transforms.CenterCrop(image_size*2),
#     transforms.Resize(image_size, interpolation=PIL.Image.NEAREST),
# ])

seg_trans_totensor = transforms.Compose([
    transforms.CenterCrop(680),
    transforms.Resize(image_size, interpolation=PIL.Image.NEAREST),
])

out_path_prefix = '../data/replica/Replica/'
data_root = lambda x: '/media/hdd/Replica-Dataset/vmap/{}/imap/00'.format(x)
# scenes = ['scene0050_00', 'scene0084_00', 'scene0580_00', 'scene0616_00']
scenes = ['room_0', 'room_1', 'room_2', 'office_0', 'office_1', 'office_2', 'office_3', 'office_4']
out_names = ['scan1', 'scan2', 'scan3', 'scan4', 'scan5', 'scan6', 'scan7', 'scan8']

background_cls_list = [5,12,30,31,40,60,92,93,95,97,98,79] 
# merge more class into background
background_cls_list.append(37) # door
# background_cls_list.append(0) # undefined: 0 for this class, we mannully organize the data and the result can be found in each instance-mapping.txt
background_cls_list.append(56) # panel
background_cls_list.append(62) # pipe

for scene, out_name in zip(scenes, out_names):
# for scene, out_name in zip()
    out_path = os.path.join(out_path_prefix, out_name)
    os.makedirs(out_path, exist_ok=True)
    print(out_path)

    # folders = ["image", "mask", "depth", "segs"]
    folders = ['segs']
    for folder in folders:
        out_folder = os.path.join(out_path, folder)
        os.makedirs(out_folder, exist_ok=True)

    # process segmentation
    segs_path = os.path.join(data_root(scene), 'semantic_instance')
    segs_paths = sorted(glob.glob(os.path.join(segs_path, 'semantic_instance_*.png')),
        key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))
    print(segs_paths)

    labels_path = os.path.join(data_root(scene), 'semantic_class')
    labels_paths = sorted(glob.glob(os.path.join(labels_path, 'semantic_class_*.png')),
        key = lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))
    print(labels_paths)
    
    instance_semantic_mapping = {}
    instance_label_mapping = {}
    label_instance_counts = {}

    out_index = 0
    for idx, (seg_path, label_path) in enumerate(zip(segs_paths, labels_paths)):
        if idx % 20 !=0: continue
        print(idx, seg_path, label_path)
        # save segs
        target_image = os.path.join(out_path, 'segs', '{:06d}_segs.png'.format(out_index))
        print(target_image)
        seg = Image.open(seg_path)
        seg_tensor = seg_trans_totensor(seg)
        seg_tensor.save(target_image)

        # label_mapping
        label_np = cv2.imread(label_path, cv2.IMREAD_UNCHANGED).astype(np.int32).transpose(1,0)
        # if 30 in np.unique(_np):
        #     import pdb; pdb.set_trace()
        # label_np = label_np.

        segs_np = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED).astype(np.int32).transpose(1,0)
        insts = np.unique(segs_np)
        for inst_id in insts:
            inst_mask = segs_np == inst_id 
            sem_cls = int(np.unique(label_np[inst_mask]))
            # import pdb; pdb.set_trace()
            instance_semantic_mapping[inst_id] = sem_cls
            if sem_cls in background_cls_list:
                instance_semantic_mapping[inst_id] = 0
                instance_label_mapping[inst_id] = 0
            # assert sem_cls.shape[0]!=0
            elif sem_cls in label_instance_counts:
                if inst_id not in instance_label_mapping:
                    inst_count = label_instance_counts[sem_cls] + 1
                    label_instance_counts[sem_cls] = inst_count
                    # chaneg the instance label mapping index to 100*label + inst_count
                    instance_label_mapping[inst_id] = sem_cls * 100 + inst_count
                else:
                    continue # already saved
            else:
                inst_count = 1
                label_instance_counts[sem_cls] = inst_count
                instance_label_mapping[inst_id] = sem_cls * 100 + inst_count

        out_index += 1

    # save the instance mapping file to output path 
    print({k: v for k, v in sorted(instance_label_mapping.items())})
    with open(os.path.join(out_path, 'instance_mapping_new.txt'), 'w') as f:
        # f.write(str(sorted(label_set)).strip('[').strip(']'))
        for k, v in instance_label_mapping.items():
            # instance_id, semantic_label, updated_instance_label (according to the number in this semantic class)
            f.write(str(k)+','+str(instance_semantic_mapping[k])+','+str(v)+'\n')
    f.close()

    #np.savez(os.path.join(out_path, "cameras_sphere.npz"), **cameras)
    # np.savez(os.path.join(out_path, "cameras.npz"), **cameras)
