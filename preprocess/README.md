# Data Preprocess
Here we privode script to preprocess the data if you start from scratch. Take the ScanNet dataset as a example. You can download one scene from ScanNet dataset, including RGB images, campose poses/intrinsics and semantic/instance segmentations. Please refer to [ScanNet dataformat](https://github.com/ScanNet/ScanNet#data-organization) and [here](https://github.com/ScanNet/ScanNet/tree/master/SensReader/python) for more details.


First, you need to prerun the Omnidata model (please install [omnidata model](https://github.com/EPFL-VILAB/omnidata) before running the command) to predict monocular cues for image. Note that the Omnidata is trained in 384*384, we follow MonoSDF to apply center-crop on original image to extract these information.
```
cd preprocess
python extract_monocular_cues.py --task depth --img_path PATH_to_your_image --output_path PATH_to_SAVE --omnidata_path YOUR_OMNIDATA_PATH --pretrained_models PRETRAINED_MODELS
python extract_monocular_cues.py --task normal --img_path PATH_to_your_image --output_path PATH_to_SAVE --omnidata_path YOUR_OMNIDATA_PATH --pretrained_models PRETRAINED_MODELS
```

Now you will get the monocular supervision. Then you need to organize the dataset to make it ready for training. 

```
python scannet_to_objsdfpp.py
```

You can perform similar opeartion on Replica dataset to get the instance label mapping file.
```
python replica_to_objsdfpp.py
```

Here are some notes:
1.  We merge some instance classes (such as ceiling, wall...) into the background. You can edit the `instance_mapping.txt` to define the objects you want.
2.  The center and scale paramters mainly used for normalize the entire scene into a cube box. It is widely-used in many NeRF projects to obtain the camera poses for training.
3.  We assume that the mask is view-consistent (i.e, the index of instance will not change). This can be done by a front-end segmentation algorithm (e.g. a video segmentation model).

Please check these scripts for more details.