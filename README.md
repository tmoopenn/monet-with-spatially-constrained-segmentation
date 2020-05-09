# MoNet with Temporal Information and Spatially-Constrained Semantic Segmentation
This project provides a Pytorch implementation of [MONet](https://arxiv.org/abs/1901.11390) with proposed extensions for 
incorporating temporal information across sequential frames in applicable domains (eg. games, videos) as well
placing additional spatial constraints on produced segmentations to improve their spatial consistency.

Base implementation of MoNet is provided by the project [MONet-pytorch](https://github.com/baudm/MONet-pytorch).

## Setup Details 
This repository is developed and tested with python 3.7.3 on Ubuntu18.04. If you are missing python header files, install python 3.7-dev

Run the setup script:
```bash   
./setup.sh
```

## Train
By default, models are configured for running on CUDA GPUs. To run on CPU, set the gpu_ids option to -1 as below:
```bash
python train.py --gpu_ids -1 
```

### Train on CLEVR Dataset 
- Download a CLEVR dataset:
```bash
wget -cN https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
```
- Train a model:
```bash
python train.py --gpu_ids -1 --batch_size 50 --dataroot /path/to/clevr_dataset --dataset_mode clevr  --num_slots 4
```

### Train on Atari Dataset 
For training on Atari, a static datset can be used as input or batches can be generated dynamically
from a random or pretrained agent.  

#### Train on Atari Dataset with Dynamic Batch Generation 
Run the train command with the following options:
```bash
# Collect frames from episode rollouts executed with a pretrained ppo agent 
python train.py --game RiverraidNoFrameskip-v4 --dynamic_datagen --collect_mode pretrained_ppo

# Collect frames from episode rollouts executed with a random agent 
python train.py --game RiverraidNoFrameskip-v4 --dynamic_datagen --collect_mode random_agent 

```

#### Train on Atari Dataset with Static Dataset 
Run the train commmand with the following options:
```bash
python train.py --dataroot /path/to/atari_dataset
```
Note: See [atari-representation-learning](https://github.com/tmoopenn/atari-representation-learning) repository for generating frames to build an Atari dataset.
[generate_atari_frames.py](https://github.com/tmoopenn/atari-representation-learning/blob/master/generate_atari_frames.py) contains examples for generating a dataset.

## Visualizing Training
We utilize Visdom for producing visuals during training. Launch the visdom server before starting the training. By default, the server is hosted on port 8097.  
```bash
python -m visdom.server
```

