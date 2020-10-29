import mmcv
import numpy as np
import torch
from utils.image_utils import imnormalize
from PIL import Image
import math

from utils.image_utils import imrescale


class ImageTransform(object):
    """Preprocess an image.

    1. rescale/resize the image to expected size
    2. normalize the image
    3. flip the image (if needed)
    4. pad the image (if needed)
    5. transpose to (c, h, w)
    """

    def __init__(
            self,
            mean=(0, 0, 0),
            std=(1, 1, 1),
            to_rgb=True,
    ):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, img, scale, flip=False, pad_val=(0, 0, 0), keep_ratio=True):

        """

        :param img:
        :param scale: (w, h)
        :param flip:
        :param pad_val:
        :param keep_ratio:
        :return:
        """

        # 1. rescale/resize the image to expected size
        if keep_ratio:
            # Resize image while keeping the aspect ratio.
            # The image will be rescaled as large as possible within the scale.
            img, scale_factor = imrescale(
                img=img,
                scale=scale,
                return_scale=True,
                interpolation='bilinear',
            )
        else:
            # Resize image to a given size ignoring the aspect ratio.
            img, w_scale, h_scale = mmcv.imresize(
                img=img,
                size=scale,
                return_scale=True,
                interpolation='bilinear',
            )
            scale_factor = np.array(
                [w_scale, h_scale, w_scale, h_scale], dtype=np.float32
            )

        # 2. normalize the image
        img_shape = img.shape
        img = imnormalize(img, self.mean, self.std, self.to_rgb)

        # 3. flip the image (if needed)
        if flip:
            img = mmcv.imflip(img)

        # 4. pad the image to the exact scale value
        if img_shape != scale:
            img = mmcv.impad(img=img, shape=scale if isinstance(scale, (int, float)) else (scale[1], scale[0]), pad_val=pad_val)
            pad_shape = img.shape
        else:
            pad_shape = img_shape

        # 5. transpose to (c, h, w)
        img = img.transpose(2, 0, 1)

        return img, img_shape, pad_shape, scale_factor


def image_transfer_back(img, scale, cur_shape, ori_shape):
    """
    numpy function. tensors not supported.
    :param img:
    :param scale:
    :param cur_shape:
    :param ori_shape:
    :return:
    """
    # (H, W)
    if img.shape == cur_shape:
        img = np.array(
            Image.fromarray(img).resize(
                (int(math.ceil(cur_shape[1] / scale)), int(math.ceil(cur_shape[0] / scale))),
                Image.BILINEAR,
            )
        )

        img = img[0:ori_shape[0], 0:ori_shape[1]]
        return img

    else:
        raise RuntimeError('img shape order not supported')


def _bbox_flip(bboxes, img_shape):
    """Flip bboxes horizontally.

    Args:
        bboxes(ndarray): shape (..., 4*k)
        img_shape(tuple): (height, width)
    """
    assert bboxes.shape[-1] % 4 == 0
    w = img_shape[1]
    flipped = bboxes.copy()
    flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
    flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
    return flipped


class BboxTransform(object):
    """Preprocess gt bboxes.

    1. rescale bboxes according to image size
    2. flip bboxes (if needed)
    3. pad the first dimension to `max_num_gts` (all images have the same number of boxes)
    """

    def __init__(self, max_num_gts=None):
        self.max_num_gts = max_num_gts

    def __call__(self, bboxes, img_shape, scale_factor, flip=False):
        gt_bboxes = bboxes * scale_factor

        if flip:
            gt_bboxes = _bbox_flip(gt_bboxes, img_shape)

        gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1] - 1)
        gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0] - 1)

        if self.max_num_gts is None:
            return gt_bboxes
        else:
            num_gts = gt_bboxes.shape[0]
            padded_bboxes = np.zeros((self.max_num_gts, 4), dtype=np.float32)
            padded_bboxes[:num_gts, :] = gt_bboxes
            return padded_bboxes


class MaskTransform(object):
    """Preprocess masks.

    1. resize masks to expected size and stack to a single array
    2. flip the masks (if needed)
    3. pad the masks (if needed)
    """

    def __call__(self, masks, pad_shape, scale_factor, flip=False):
        # Resize image while keeping the aspect ratio.
        # The image will be rescaled as large as possible within the scale.
        masks = [
            imrescale(mask, scale_factor, interpolation='nearest')
            for mask in masks
        ]

        if flip:
            masks = [mask[:, ::-1] for mask in masks]

        padded_masks = [
            mmcv.impad(mask, pad_shape[:2], pad_val=0) for mask in masks
        ]

        padded_masks = np.stack(padded_masks, axis=0)
        return padded_masks


class SegMapTransform(object):
    """Preprocess semantic segmentation maps.

    1. rescale the segmentation map to expected size
    3. flip the image (if needed)
    4. pad the image (if needed)
    """

    def __init__(self, size_divisor=None):
        self.size_divisor = size_divisor

    def __call__(self, img, scale, flip=False, keep_ratio=True):
        if keep_ratio:
            img = imrescale(img, scale, interpolation='nearest')
        else:
            img = mmcv.imresize(img, scale, interpolation='nearest')

        if flip:
            img = mmcv.imflip(img)

        if self.size_divisor is not None:
            img = mmcv.impad_to_multiple(img, self.size_divisor)

        return img


class Numpy2Tensor(object):

    def __init__(self):
        pass

    def __call__(self, *args):
        if len(args) == 1:
            return torch.from_numpy(args[0])
        else:
            return tuple([torch.from_numpy(np.array(array)) for array in args])
