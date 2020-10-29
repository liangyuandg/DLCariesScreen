from __future__ import division

from functools import partial
import torch
import argparse
import logging
import random
import numpy as np
import mmcv
from collections import OrderedDict
torch.autograd.set_detect_anomaly(True)

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from detectors.SSD_VGG16_Detector import SSDDetector
from datasets.DentalDataset import DentalDataset
from utils.sampler import DistributedNonGroupSampler
from mmcv.parallel.collate import collate

import warnings
warnings.simplefilter("error")
warnings.simplefilter("ignore", DeprecationWarning)


from mmcv.runner import Runner, DistSamplerSeedHook
from utils.distribution import DistOptimizerHook

# dataset
IMG_PREFIX = '/PATH/TO/REPO/OF/IMAGES/'
ANN_FILE = '/PATH/TO/PICKLE/THAT/INDICATES/TRAINING/IMAGES/train.pickle'

# image
NUM_CLASS = 2
IMG_SCALE = (300, 300)
FLIP = True
FLIP_RATIO = 0.5
IMG_NORM_CFG = \
    dict(
        mean=[123.675, 116.28, 103.53],
        std=[128.0, 128.0, 128.0],
        to_rgb=True,
        pad_values=(0.0, 0.0, 0.0),
        resize_keep_ratio=True,
    )

EXTRA_AUG = dict(
    photo_metric_distortion=dict(
        brightness_delta=20,
        contrast_range=(0.5, 1.3),
        saturation_range=(0.5, 1.5),
        hue_delta=10,
        swap_channel=False,
    ),
    expand=dict(
        mean=IMG_NORM_CFG['mean'],
        to_rgb=IMG_NORM_CFG['to_rgb'],
        ratio_range=(1, 3)
    ),
    random_crop=dict(
        min_ious=(0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.5
    )
)
# EXTRA_AUG = None

# log
LOG_LEVEL = 'INFO'
LOG_INTERVAL = 1  # 20 iters
log_config = dict(
    interval=LOG_INTERVAL,  # 50 iters
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ]
)

# read and save modell
WORK_DIR = '/PATH/TO/WORKSPACE/DIRECTORY/FOR/STORING/TRAINED/MODELS/'
# model pre-trained on ImageNet for quick initialization
LOAD_FROM = '/PATH/TO/DLCariesScreen/checkpoints/vgg16-397923af_0321_standardSSD.pth'

# training config
SEED = None
DO_VALIDATION = False
TOTAL_EPOCHS = 600

# loading
WORKERS_PER_GPU = 8
IMGS_PER_GPU = 10

# optimizer: SGD
LR = 5e-4
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[16, 22]
)
grad_clip = dict(max_norm=35, norm_type=2)
workflow = [('train', 1)]
total_epochs = TOTAL_EPOCHS

# checkpoints saving. interval for how many epoch.
CHECKPOINT_INTERVAL = 20
checkpoint_config = dict(interval=CHECKPOINT_INTERVAL)

# set True when input size does not vary a lot
torch.backends.cudnn.benchmark = True
# train head part only 
TRAIN_HEAD_ONLY = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--world_size', type=int)
    args = parser.parse_args()
    print('what is the rank of the current program: ')
    print(args.local_rank)
    print('world size: ')
    print(args.world_size)

    # initialize dist
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    torch.cuda.set_device(int(args.local_rank))
    dist.init_process_group(backend='nccl', init_method='env://')

    # init logger before other steps
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=LOG_LEVEL
        )
    if args.local_rank != 0:
        logger.setLevel('ERROR')
    logger.info('Starting Distributed training')

    # set random seeds
    if SEED is not None:
        logger.info('Set random seed to {}'.format(SEED))
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    # build dataset
    dental_dataset = DentalDataset(
        num_class=NUM_CLASS,
        ann_file=ANN_FILE,
        img_prefix=IMG_PREFIX,
        img_scale=IMG_SCALE,
        img_norm_cfg=IMG_NORM_CFG,
        multiscale_mode='value', # select a scale, rather than random from a range.
        flip_ratio=FLIP_RATIO,
        with_label=True,
        extra_aug=EXTRA_AUG,
        test_mode=False,
    )

    # build model
    model = SSDDetector(
        # basic
        input_size=IMG_SCALE,
        num_classes=NUM_CLASS,
        in_channels=(512, 1024, 512, 256, 256),
        use_dropout=False,
        dropout_rate=None,
        # anchor generate
        anchor_ratios=([1 / 2.0, 1.0, 2.0], [1/3.0, 1 / 2.0, 1.0, 2.0, 3.0], [1 / 3.0, 1 / 2.0, 1.0, 2.0, 3.0], [1 / 3.0, 1 / 2.0, 1.0, 2.0, 3.0], [1 / 2.0, 1.0, 2.0]),
        anchor_strides=((16, 16), (16, 16), (30, 30), (60, 60), (100, 100)),
        basesizes=((12, 12), (16, 16), (24, 24), (30, 30), (36, 36)),
        allowed_border=-1,
        # regression
        target_means=(.0, .0, .0, .0),
        target_stds=(0.1, 0.1, 0.2, 0.2),
        # box assign
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.,
        gt_max_assign_all=False,
        # sampling
        sampling=False,
        # balancing the loss
        neg_pos_ratio=3,
        # loss
        smoothl1_beta=1.,
        # inference nms
        nms_pre=-1,
        score_thr=0.02,
        min_size=100.0,
        max_scale_ratio=10.0,
        nms_cfg=['nms', 0.45, None],
        max_per_img=200,
        # device
        device=None,
    )
    model.cuda(args.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True
    )
    if hasattr(model, 'module'):
        model = model.module
        model.device_ids = [args.local_rank, ]

    # build sampler for shuffling, padding, and mixing.
    sampler = DistributedNonGroupSampler(
        dataset=dental_dataset,
        samples_per_gpu=IMGS_PER_GPU,
        num_replicas=args.world_size,
        rank=args.local_rank,
        shuffle=True,
    )
    # build data loader.
    data_loader = DataLoader(
        dataset=dental_dataset,
        batch_size=IMGS_PER_GPU,
        # shuffle should be False when sampler is given.
        shuffle=False,
        sampler=sampler,
        batch_sampler=None,
        num_workers=WORKERS_PER_GPU,
        collate_fn=partial(collate, samples_per_gpu=IMGS_PER_GPU),
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    )

    # optimizer
    if TRAIN_HEAD_ONLY:
        train_params = []
        for name, param in model.named_parameters():
            if 'backbone.features.30' in name or 'backbone.features.32' in name or 'backbone.extra' in name or 'bbox_head' in name:
                param.requires_grad = True
                train_params.append(param)
            else:
                param.requires_grad = False
    else:
        train_params = model.parameters()
    optimizer = torch.optim.SGD(
        params=train_params,
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    # initial folder
    if mmcv.is_str(WORK_DIR):
        mmcv.mkdir_or_exist(WORK_DIR)
    else:
        raise TypeError('"work_dir" must be a str or None')

    model.train()

    # plan B
    runner = Runner(
        model=model,
        batch_processor=batch_processor,
        optimizer=optimizer,
        work_dir=WORK_DIR,
        log_level=logging.INFO,
        logger=None,
    )
    # register hooks: optimization after the forward
    optimizer_config = DistOptimizerHook(
        grad_clip=grad_clip,
        coalesce=True,
        bucket_size_mb=-1,
    )
    # register hooks: along with training
    runner.register_training_hooks(
        lr_config=lr_config,
        optimizer_config=optimizer_config,
        checkpoint_config=checkpoint_config,
        log_config=log_config
    )
    # register hooks: set sampler seed before each epoch
    runner.register_hook(DistSamplerSeedHook())
    if LOAD_FROM is not None:
        runner.load_checkpoint(LOAD_FROM, strict=False)
    runner.run(data_loaders=[data_loader], workflow=workflow, max_epochs=total_epochs)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def batch_processor(model, data, train_mode):
    losses = model(
        img=data['img'].data[0].cuda(model.device_ids[0], non_blocking=True),
        img_meta=data['img_meta'].data[0],
        gt_bboxes=[item.cuda(model.device_ids[0], non_blocking=True) for item in data['gt_bboxes'].data[0]],
        gt_labels=[item.cuda(model.device_ids[0], non_blocking=True) for item in data['gt_labels'].data[0]],
        is_test=False,
    )

    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
    log_vars['loss'] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()

    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data)
    )

    return outputs


if __name__ == '__main__':
    main()
