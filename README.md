<p align="center">

  <h1 align="center">ObjectSDF++: Improved Object-Compositional Neural Implicit Surfaces</h1>
  <p align="center">
    <a href="http://qianyiwu.github.io/">Qianyi Wu</a>
    ·
    <a href="https://scholar.google.com/citations?user=2Pedf3EAAAAJ">Kaisiyuan Wang</a>
    ·
    <a href="https://likojack.github.io/kejieli/#/home">Kejie Li</a>
    ·
    <a href="https://scholar.google.com/citations?user=sGCf2k0AAAAJ">Jianmin Zheng</a>
    ·
    <a href="https://jianfei-cai.github.io/">Jianfei Cai</a>

  </p>
  <h3 align="center">ICCV 2023</h3>
  <h3 align="center"><a href="">Paper</a> | <a href="https://qianyiwu.github.io/objectsdf++">Project Page</a></h3>
  <div align="center"></div>
</p>

<p align="center">
  <a href="">
    <img src="./media/teaser.gif" alt="Logo" width="95%">
  </a>
</p>

<p align="center">
<strong>TL; DR:</strong> We propose an occlusion-aware opacity rendering formulation to better use the instance mask supervision. Together with an object-distinction regularization term, the proposed ObjectSDF++ produces more accurate surface reconstruction at both scene and object levels.
</p>
<br>

# Setup

## Installation
This code has been tested on Ubuntu 22.02 with torch 2.0 & CUDA 11.7 on a RTX 3090.
Clone the repository and create an anaconda environment named objsdf
```
git clone https://github.com/QianyiWu/objectsdf_plus.git
cd objectsdf_plus

conda create -y -n objsdf python=3.9
conda activate object

pip install -r requirements.txt
```
The hash encoder will be compiled on the fly when running the code.

## Dataset
For downloading the preprocessed data, run the following script. The data for the Replica and ScanNet is adapted from [MonoSDF](https://github.com/autonomousvision/monosdf), [vMAP](https://github.com/kxhit/vMAP).
```
bash scripts/download_dataset.sh
```
# Training

Run the following command to train ObjectSDF++:
```
cd ./code
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 training/exp_runner.py --conf CONFIG  --scan_id SCAN_ID
```
where CONFIG is the config file in `code/confs`, and SCAN_ID is the id of the scene to reconstruct.

We provide example commands for training Replica dataset as follows:
```
# Replica scan 1 (room0)
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 training/exp_runner.py --conf confs/replica_objsdfplus.conf --scan_id 1

# ScanNet scan 1 (scene_0050_00)
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 training/exp_runner.py --conf confs/scannet_objsdfplus.conf --scan_id 1

```
The intermediate results and checkpoints will be saved in ``exps`` folder. 

# Evaluations

## Replica
Evaluate one scene (take scan 1 room0 for example)
```
cd replica_eval
python evaluate_single_scene.py --input_mesh replica_scan1_mesh.ply --scan_id 1 --output_dir replica_scan1
```

We also provided scripts for evaluating all Replica scenes and objects:
```
cd replica_eval
python evaluate.py  # scene-level evaluation
python evaluate_3D_obj.py   # object-level evaluation
```
please check the script for more details. For obtaining the object groundtruth, you can refer to [here](https://github.com/kxhit/vMAP#dataset) for more details.

## ScanNet
```
cd scannet_eval
python evaluate.py
```
please check the script for more details.

# Acknowledgements
This project is built upon [MonoSDF](https://github.com/autonomousvision/monosdf). The monocular depth and normal images are obtained by [Omnidata](https://omnidata.vision). The evaluation of object reconstruction is inspired by [vMAP](https://github.com/kxhit/vMAP). Cuda implementation of Multi-Resolution hash encoding is heavily based on [torch-ngp](https://github.com/ashawkey/torch-ngp). Kudos to these researchers.


# Citation
If you find our code or paper useful, please cite the series of ObjectSDF works.
```BibTeX
@inproceedings{wu2022object,
  title      = {Object-compositional neural implicit surfaces},
  author     = {Wu, Qianyi and Liu, Xian and Chen, Yuedong and Li, Kejie and Zheng, Chuanxia and Cai, Jianfei and Zheng, Jianmin},
  booktitle  = {European Conference on Computer Vision},
  year       = {2022},
}

@inproceedings{wu2023objsdfplus,
  author    = {Wu, Qianyi and Wang, Kaisiyuan and Li, Kejie and Zheng, Jianmin and Cai, Jianfei},
  title     = {ObjectSDF++: Improved Object-Compositional Neural Implicit Surfaces},
  booktitle = {ICCV},
  year      = {2023},
}
```

