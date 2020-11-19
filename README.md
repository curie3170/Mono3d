# Mono3D

## Installation
```bash
# cloning the repo.
git clone https://github.com/Uehwan/mono3d
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
CUDA_VISIBLE_DEVICES=0,1 python train_pixor_e2e.py -c configs/fusion.cfg --depth_pretrain <depth_model_pretrain_path> --pixor_pretrain <detection_model_pretrain_path>
```

## Evaluation

## Performance

## License

## Citation

## Acknowledgement
