import argparse
import torch
import mmcv
from functools import partial

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from datasets.DentalDataset import DentalDataset
from detectors.SSD_VGG16_Detector import SSDDetector
from mmcv.parallel.collate import collate
from utils.sampler import NewDistributedSampler
from utils.inference import collect_results

# checkpoints
CHECKPOINT_FILE = '/PATH/TO/CHECKPOINT'
# output
# tmp dir for writing some results
TMPDIR = '/PATH/TO/A/TEMPORARY/DIRECTORY/'
# final result dir
OUT_FILE = '/PATH/TO/DIRECTORY/TO/SAVE/RESULTS/results.pickle'
# input
IMG_PREFIX = '/PATH/TO/REPO/OF/IMAGES/'
ANN_FILE = '/PATH/TO/PICKLE/THAT/INDICATES/TESTING/IMAGES/test.pickle'

# image
NUM_CLASS = 2
IMG_SCALE = (300, 300)
FLIP = False
FLIP_RATIO = 0
IMG_TRANSFORM_CONFIG = \
    dict(
        mean=[123.675, 116.28, 103.53],
        std=[128.0, 128.0, 128.0],
        to_rgb=True,
        pad_values=(0, 0, 0),
        resize_keep_ratio=True,
    )

# loading
WORKERS_PER_GPU = 8
IMGS_PER_GPU = 12

# set True when input size does not vary a lot
torch.backends.cudnn.benchmark = True


def main():

    # get local rank from distributed launcher
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
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

    # define dataset
    dataset = DentalDataset(
        num_class=NUM_CLASS,
        ann_file=ANN_FILE,
        img_prefix=IMG_PREFIX,
        img_scale=IMG_SCALE,
        img_norm_cfg=IMG_TRANSFORM_CONFIG,
        multiscale_mode='value',
        flip_ratio=FLIP_RATIO,
        with_label=False,
        extra_aug=None,
        test_mode=True,
    )

    # sampler for make number of samples % number of gpu == 0
    sampler = NewDistributedSampler(
        dataset=dataset,
        num_replicas=args.world_size,
        images_per_gpu=IMGS_PER_GPU,
        rank=args.local_rank,
        shuffle=False
    )

    # data loader. Note this is the code for one (each) gpu.
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=IMGS_PER_GPU,
        # when sampler is given, shuffle must be False.
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

    # define the model and restore checkpoint
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
        output_device=args.local_rank
    )
    if hasattr(model, 'module'):
        model = model.module

    # load checkpoint
    loc = 'cuda:{}'.format(args.local_rank)
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=loc)
    # optimizer.state_dict -> state, param_groups
    # state -> var series number -> step / exp_avg / exp_avg_sq
    # param_groups -> lr / betas / eps / weight_decay / amsgrad / params
    model.load_state_dict(checkpoint['state_dict'], strict=True)

    # enable dropout during inference
    model.eval()
    # for m in model.modules():
    #     if m.__class__.__name__.startswith('Dropout'):
    #         m.train()

    # results and progress bar
    results = []
    if args.local_rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    # enumerate all data
    for i, data_pair in enumerate(data_loader):
        data_pair_img = data_pair['img'].data[0].cuda(args.local_rank, non_blocking=True)
        data_pair_img_meta = data_pair['img_meta'].data[0]

        with torch.no_grad():
            result = model(
                is_test=True,
                img=data_pair_img,
                img_meta=data_pair_img_meta,
                rescale=True,
            )
        results.extend(result)

        # update program bar only if it is rank 0.
        if args.local_rank == 0:
            for _ in range(IMGS_PER_GPU * args.world_size):
                prog_bar.update()

    # collect results from all gpus
    results = collect_results(
        result_part=results,
        dataset_real_size=len(dataset),
        tmpdir=TMPDIR
    )

    # write results to file
    # [Number of images, Number of classes, (k, 5)].
    # 5 for t, l, b, r, and prob.
    if args.local_rank == 0:
        print('\nwriting results to {}'.format(OUT_FILE))
        mmcv.dump(results, OUT_FILE)


if __name__ == '__main__':
    main()
