# This code is based on https://github.com/blackzxy/logah and has been modified for this project.
# Please refer to the original repository for the foundational implementation and citation.
#
# Copyright (c) 2023. Samsung Electronics Co., Ltd. All Rights Reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Trains a TAL on the ImageNet. DistributedDataParallel (DDP) training is
used if `torchrun` is used. This script assumes the Decathlon dataset is already downloaded and set up as required.

Examples:

Architecture dataset: ViTs+-1K
This trains TAL+ on the ImageNet:

python train_ghn_ddp_MTL_teacher.py \
    -n -v 50 --ln -e 75 --amp -m 1 \
    --name ghn-tal+-imagenet \
    -d imagenet --data_dir /datadata/imagenet \
    --batch_size 256 --hid 128 --lora_r 90 --layers 5 --heads 16 \
    --opt adamw --lr 0.3e-3 --wd 1e-2 \
    --scheduler cosine-warmup --debug 0 --max_shape 2048 --lora \

Architecture dataset: ViTs-1K
This trains TAL on the ImageNet:

python train_ghn_ddp_MTL_teacher.py \
    -n -v 50 --ln -e 75 --amp -m 1 \
    --name ghn-tal-imagenet \
    -d imagenet --data_dir /data/imagenet \
    --batch_size 512 --hid 128 --lora_r 90 --layers 5 --heads 16 \
    --opt adamw --lr 0.3e-3 --wd 1e-2 \
    --scheduler cosine-warmup --debug 0 --max_shape 2048 --lora \

"""


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
import argparse
import torch.distributed as dist
from functools import partial
from config import init_config
import ema as ema
from ppuda.vision.loader import image_loader
from ghn3.vit1m import DeepNets1MDDP
from ghn3.nn import GHN3
from ghn3.utils import log
from ghn3.trainer_teacher import Trainer
from ghn3.ddp_utils import setup_ddp, clean_ddp
import timm
import time
log = partial(log, flush=True)

# Define paths for open source use. Please set these paths as needed.
CKPT_PATH = './pretrained/vit_base_patch16_224.orig_in21k_ft_in1k.bin'  # Path to teacher model checkpoint
DATA_DIR = './data/imagenet'  # Path to your ImageNet

def main():
    parser = argparse.ArgumentParser(description='GHN-3 training')
    parser.add_argument('--heads', type=int, default=8, help='number of self-attention heads in GHN-3')
    parser.add_argument('--compile', type=str, default=None, help='use pytorch2.0 compilation for potential speedup')
    parser.add_argument('--ghn2', action='store_true', help='train GHN-2, also can use code from'
                                                            ' https://github.com/facebookresearch/ppuda to train GHN-2')
    parser.add_argument('--interm_epoch', type=int, default=5, help='intermediate epochs to keep checkpoints for')
    ghn2 = parser.parse_known_args()[0].ghn2

    ddp = setup_ddp()
    args = init_config(mode='train_ghn', parser=parser, verbose=ddp.rank == 0,
                       debug=0,   # to avoid extra sanity checks and make training faster
                       layers=3,  # default number of layers in GHN-3
                       shape_multiplier=2 if ghn2 else 1)  # max_shape default setting (can be overriden by --max_shape)

    if hasattr(args, 'multigpu') and args.multigpu:
        raise NotImplementedError(
            'the `multigpu` argument was meant to use nn.DataParallel in the GHN-2 code. '
            'nn.DataParallel is likely to be deprecated in PyTorch in favor of nn.DistributedDataParallel '
            '(https://github.com/pytorch/pytorch/issues/659360).'
            'Therefore, this repo is not supporting DataParallel anymore as it complicates some steps. '
            'nn.DistributedDataParallel is used if this script is called with torchrun (see examples on top).')
    is_imagenet = args.dataset.startswith('imagenet')

    log('loading the %s dataset...' % args.dataset.upper())
    train_queue, _, num_classes = image_loader(args.dataset,
                                               args.data_dir,
                                               im_size=args.imsize,
                                               test=False,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               seed=args.seed,
                                               verbose=ddp.rank == 0)

    hid = args.hid
    s = 16 if is_imagenet else 11
    dmax_shape = 2048
    #default_max_shape = (hid * 2, hid * 2, s, s) if ghn2 else (hid, hid, s, s)
    default_max_shape = (dmax_shape, dmax_shape, s, s) if ghn2 else (dmax_shape, dmax_shape, s, s)
    log('current max_shape: {} {} default max_shape: {}'.format(args.max_shape,
                                                                '=' if args.max_shape == default_max_shape else '!=',
                                                                default_max_shape))

    config = {'max_shape': args.max_shape, 'num_classes': num_classes, 'hypernet': args.hypernet,
              'lora': args.lora, 'lora_r': args.lora_r, 'max_ck_lora': args.max_ck_lora, 'use_1d_decoder': args.use_1d_decoder,
              'decoder': args.decoder, 'weight_norm': args.weight_norm, 've': args.virtual_edges > 1,
              'layernorm': args.ln, 'hid': hid, 'layers': args.layers, 'heads': args.heads, 'is_ghn2': ghn2}

    ghn = GHN3(**config, debug_level=args.debug).to(args.device)
    ema_helper = None
    ### Apply EMA ###
    if args.ema:
        ema_helper = ema.EMAHelper(mu = args.ema_rate)
        ema_helper.register(ghn)
    ### Apply EMA ###

    graphs_queue, sampler = DeepNets1MDDP.loader(args.meta_batch_size // (ddp.world_size if ddp.ddp else 1),
                                                 dense=ghn.is_dense(),
                                                 wider_nets=is_imagenet,
                                                 split=args.split,
                                                 nets_dir=args.data_dir,
                                                 virtual_edges=args.virtual_edges,
                                                 num_nets=args.num_nets,
                                                 large_images=is_imagenet,
                                                 verbose=ddp.rank == 0,
                                                 debug=args.debug > 0)

    # load teachermodel
    teachermodel = timm.create_model("timm/vit_base_patch16_224.orig_in21k_ft_in1k",pretrained=False)
 
    teacher_checkpoint_path = CKPT_PATH
    checkpoint = torch.load(teacher_checkpoint_path, map_location='cpu')  # Use map_location='cpu' if running on CPU

    teachermodel.load_state_dict(checkpoint)
    
    
    trainer = Trainer(ghn,
                      teachermodel,
                      opt=args.opt,
                      opt_args={'lr': args.lr, 'weight_decay': args.wd, 'momentum': args.momentum},
                      scheduler='mstep' if args.scheduler is None else args.scheduler,
                      scheduler_args={'milestones': args.lr_steps, 'gamma': args.gamma},
                      n_batches=len(train_queue),
                      grad_clip=args.grad_clip,
                      device=args.device,
                      log_interval=args.log_interval,
                      amp=args.amp,
                      amp_min_scale=1024,       # this helped stabilize AMP training
                      amp_growth_interval=100,  # this helped stabilize AMP training
                      predparam_wd=0 if ghn2 else 3e-5,
                      label_smoothing=0.1 if is_imagenet else 0.0,
                      save_dir=args.save,
                      ckpt=args.ckpt,
                      epochs=args.epochs,
                      verbose=ddp.rank == 0,
                      compile_mode=args.compile,
                      ema=args.ema,
                      ema_helper=ema_helper,
                      )

    log('\nStarting training GHN with {} parameters!'.format(sum([p.numel() for p in ghn.parameters()])))
    if ddp.ddp:
        # make sure sample order is different for each seed
        sampler.sampler.seed = args.seed
        log(f'shuffle DeepNets1MDDP train loader: set seed to {args.seed}')
        # for each DeepNets1MDDP epoch, the graph loader will be shuffled inside the ghn3/deepnets1m.py

    graphs_queue = iter(graphs_queue)

    start_time = time.time()
    for epoch in range(trainer.start_epoch, args.epochs):
        

        log('\nepoch={:03d}/{:03d}, lr={:e}'.format(epoch + 1, args.epochs, trainer.get_lr()))

        trainer.reset_metrics(epoch)

        for step, (images, targets) in enumerate(train_queue, start=trainer.start_step):

            if step >= len(train_queue):  # if we resume training from some start_step > 0, then need to break the loop
                break

            trainer.update(images, targets, graphs=next(graphs_queue))
            trainer.log(step)

            if args.save:
                trainer.save(epoch, step, {'args': args, 'config': config}, interm_epoch=args.interm_epoch)

        trainer.scheduler_step()  # lr scheduler step

    log('done at {}!'.format(time.strftime('%Y%m%d-%H%M%S')))
    if ddp.ddp:
        clean_ddp()

    end_time = time.time()
    log('Training took {:.2f} mins'.format((end_time - start_time) / 60))

if __name__ == '__main__':
    main()