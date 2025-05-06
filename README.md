# Learngene Tells You How to Customize: Task-Aware Parameter Initialization at Flexible Scales


## Introduction


We propose Task-Aware Learngene (TAL), a novel graph hypernetwork-based method for predicting model parameters conditioned on model scale and task characteristics, enabling efficient initialization and strong transfer learning across diverse tasks.

## Requirements

You can directly clone the environment from [Parameter Prediction for Unseen Deep Architectures (PPUDA)](https://github.com/facebookresearch/ppuda) and install the required package with the following command:

```bash
pip install git+https://github.com/facebookresearch/ppuda.git
```


## Architecture Dataset Generator

We provide code for generating the ViTs-1K and ViTs+-1K  datasets.  
- `vit_generator.py` is used for generating the ViTs-1K/ViTs+-1K  dataset.


## Training TAL on ImageNet with teachermodel

```bash
python train_ghn_ddp_MTL_teacher.py \
    -n -v 50 --ln -e 75 --amp -m 1 \
    --name ghn-tal-imagenet \
    -d imagenet --data_dir /data/imagenet \
    --batch_size 512 --hid 128 --lora_r 90 --layers 5 --heads 16 \
    --opt adamw --lr 0.3e-3 --wd 1e-2 \
    --scheduler cosine-warmup --debug 0 --max_shape 2048 --lora
```


## Training TAL on Decathlon

```bash
python train_ghn_ddp_MTL_decathlon.py \
    -n -v 50 --ln -e 100 --amp -m 1 \
    --name ghn-tal+-decathlon \
    -d imagenet --data_dir /data/imagenet \
    --batch_size 512 --hid 128 --lora_r 90 --layers 5 --heads 16 \
    --opt adamw --lr 0.3e-3 --wd 1e-2 \
    --scheduler cosine-warmup --debug 0 --max_shape 2048 --lora
```

