import mmcv
import os.path as osp
import torch.distributed as dist
import shutil
from mmcv.runner import get_dist_info
from utils.image_utils import to_tensor


def _prepare_data(img, img_transform, img_scale, img_resize_keep_ratio, img_flip, device):
    ori_shape = img.shape
    img, img_shape, pad_shape, scale_factor = img_transform(
        img=img,
        scale=img_scale,
        flip=img_flip,
        keep_ratio=img_resize_keep_ratio,
    )

    img = to_tensor(img).to(device).unsqueeze(0)

    img_meta = [
        dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=img_flip
        )
    ]
    return dict(img=[img], img_meta=[img_meta])


def collect_results(result_part, dataset_real_size, tmpdir):
    """
    collect results from all gpus and concatenate them into final results.
    Note the results from paddings of dataset are removed.

    :param result_part: result from the current gpu.
    :param dataset_real_size: the real size (unpadded size) of the dataset.
    :param tmpdir: a tmpdir for saving per gpu results. will be removed latter.
    :return: ordered, unpadded results of the whole dataset.
    """
    rank, world_size = get_dist_info()

    # create a tmp dir if it is not specified
    mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    # wait for all gpus to finish.
    dist.barrier()

    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))

        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))

        # the dataloader may pad some samples
        ordered_results = ordered_results[:dataset_real_size]
        # remove tmp dir
        shutil.rmtree(tmpdir)

        return ordered_results