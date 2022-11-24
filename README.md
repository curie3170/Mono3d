# Self-supervised 3D Object Detection from Monocular Pseudo-LiDAR
#### Curie Kim, Ue-Hwan Kim, Jong-Hwan Kim
#### Accepted to IEEE MFI 2022
#### [Paper](https://arxiv.org/abs/2209.09486)

## Abstract
There have been attempts to detect 3D objects by fusion of stereo camera images and LiDAR sensor data or using LiDAR for pre-training and only monocular images for testing, but there have been less attempts to use only monocular image sequences due to low accuracy. In addition, when depth prediction using only monocular images, only scale-inconsistent depth can be predicted, which is the reason why researchers are reluctant to use monocular images alone. Therefore, we propose a method for predicting absolute depth and detecting 3D objects using only monocular image sequences by enabling end-to-end learning of detection networks and depth prediction networks. As a result, the proposed method surpasses other existing methods in performance on the KITTI 3D dataset. Even when monocular image and 3D LiDAR are used together during training in an attempt to improve performance, ours exhibit is the best performance compared to other methods using the same input. In addition, end-to-end learning not only improves depth prediction performance, but also enables absolute depth prediction, because our network utilizes the fact that the size of a 3D object such as a car is determined by the approximate size.

<p align="center"><img src="https://user-images.githubusercontent.com/17980462/177569344-01ceb000-7bd2-42d8-bf40-18e4de48b850.png"  width="100%"></p>

## Installation
```bash
# cloning the repo.
git clone https://github.com/curie3170/Mono3d.git
cd mono3d

# environment creation using conda
conda env create -f environment.yml
pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html

# compiling the kitti official evaluation code (required for the validation step)
cd evaluation/kitti_eval
./compile.sh
```

## Data Preparation

## Training
```bash
CUDA_VISIBLE_DEVICES=0,1 python train_pixor_e2e.py -c configs/fusion.cfg --root_dir <kitti_dataset_path> --depth_pretrain <depth_model_pretrain_path> --pixor_pretrain <detection_model_pretrain_path> --depth_loss <M or D or MD>
```

## Evaluation
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_pixor_e2e_unsup.py -c configs/fusion.cfg --mode eval --root_dir <kitti_dataset_path> --eval_ckpt <trained_checkpoint_path> --depth_loss <M or D or MD>
```
## Performance
<p align="center"><img src="https://user-images.githubusercontent.com/17980462/177567814-3d6d8e33-0f80-4c3f-bf7d-8ea2cb1e4fa7.png"  width="50%"></p>
<!-- <p align="center"><img src="https://user-images.githubusercontent.com/17980462/177567891-c3af00ca-e78c-431a-bf5d-f810bffbf245.png"  width="70%"></p> -->
<p align="center"><img src="https://user-images.githubusercontent.com/17980462/193173420-4684c338-cea2-41ac-9d50-b78f9481d26d.png" width="100%"></p>
<p align="center"><img src="https://user-images.githubusercontent.com/17980462/193173628-5a4375b6-9dbe-4e76-b340-e5753f2c289c.png"  width="100%"></p>

## License
This project partially depends on the sources of [Monodepth2](https://github.com/nianticlabs/monodepth2) and [PIXOR](https://github.com/philip-huang/PIXOR)

## Citation
```
@INPROCEEDINGS{9913846,  
               author={Kim, Curie and Kim, Ue-Hwan and Kim, Jong-Hwan},  
               booktitle={2022 IEEE International Conference on Multisensor Fusion and Integration for Intelligent Systems (MFI)},   
               title={Self-supervised 3D Object Detection from Monocular Pseudo-LiDAR},   
               year={2022}, 
               pages={1-6},  
               doi={10.1109/MFI55806.2022.9913846}}
```
