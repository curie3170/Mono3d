# Mono3D
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
<p align="center"><img src="https://user-images.githubusercontent.com/17980462/177567548-2b7b1e78-ccba-430a-82e8-bd3d910832c4.png"  width="100%"></p>
## License

## Citation

## Acknowledgement
