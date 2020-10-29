import mmcv
import numpy as np
from numpy import random

from cores.bbox.geometry import bbox_overlaps_np


class PhotoMetricDistortion(object):
    """brightness (Brt), contrast (Crt), Saturation (Srt), Hue (Hue) and switch channel (SW) random adjustments.
    Adjustment is done for each independently with P=0.5.
    Amount of adjustments is of uniform sampling from the given deltas.

    In RGB space:
        f(x) = Crt * (x + Brt)
        Crt < 1 reduce contrast; Crt > 1 increase contrast.
        Brt > 0 (for all channels) increase brightness; Brt < 0 reduce brightness.

    Hue (0~360): the degree to which a stimulus can be described as similar to or different
    from stimuli that are described as red (0/360), green (120), blue (240), yellow (60), etc.
    If you draw a intensity on wavelength curve for each pixel in the image,
    hue refers to the peak of the curve, i.e., at the visible wavelength with the
    greatest energy from the output.


    Saturation: the perceived intensity. In other words it is a value of
    how dominant the color is, or how colorful the object looks.
    If you draw a intensity on wavelength curve for each pixel in the image,
    saturation is the relative bandwidth of the curve.

    In HSV (Hue, Saturation, Value (brightness)) space:
        sat = Srt * sat
        hue = hue + Hue_delta

    Args:
        brightness_delta (float): [-brightness_delta, +brightness_delta].
        contrast_range (Tuple): (contrast_lower, contrast_upper)
        saturation_range (Tuple): (saturation_lower, saturation_upper)
        hue_delta (float): [-hue_delta, +hue_delta]

    Returns:
        img, boxes, labels.
    """
    def __init__(
            self,
            brightness_delta=32,
            contrast_range=(0.5, 1.5),
            saturation_range=(0.5, 1.5),
            hue_delta=18,
            swap_channel=None,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.swap_channel = swap_channel

    def __call__(self, img, boxes, labels):
        # random brightness
        if random.randint(2):
            delta = random.uniform(
                -self.brightness_delta,
                self.brightness_delta
            )
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(
                    self.contrast_lower,
                    self.contrast_upper
                )
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation: img[..., 1]
        if random.randint(2):
            img[..., 1] *= random.uniform(
                self.saturation_lower,
                self.saturation_upper
            )

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if (self.swap_channel is None or self.swap_channel is True) and random.randint(2):
            img = img[..., random.permutation(3)]

        return img, boxes, labels


class Expand(object):
    """Expand the image by RATIO with mean values padded for the rest.
    Expand the image by p=0.5, with expanding ratio from uniform sampling of ratio_range.

    Args:
        mean (Iterable): padding values.
        to_rgb (bool): in rgb space or bgr space.
        ratio_range (Tuple): [min_ratio, max_ratio]

    Returns:
        img, boxes, labels.
    """
    def __init__(self, mean=(0, 0, 0), to_rgb=True, ratio_range=(1, 4)):
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean

        self.min_ratio, self.max_ratio = ratio_range

    def __call__(self, img, boxes, labels):

        if boxes is None:
            if random.randint(2):
                return img, boxes, labels

            h, w, c = img.shape
            ratio = random.uniform(self.min_ratio, self.max_ratio)

            if -1 in self.mean:
                self.mean = np.mean(img, axis=(0, 1), dtype=img.dtype)

            expand_img = np.full(
                (int(h * ratio), int(w * ratio), c),
                self.mean
            ).astype(img.dtype)

            left = int(random.uniform(0, w * ratio - w))
            top = int(random.uniform(0, h * ratio - h))

            expand_img[top:top + h, left:left + w] = img
            img = expand_img

            return img, boxes, labels
        else:
            if random.randint(2):
                return img, boxes, labels

            h, w, c = img.shape
            ratio = random.uniform(self.min_ratio, self.max_ratio)

            if -1 in self.mean:
                self.mean = np.mean(img, axis=(0, 1), dtype=img.dtype)

            expand_img = np.full(
                (int(h * ratio), int(w * ratio), c),
                self.mean
            ).astype(img.dtype)

            left = int(random.uniform(0, w * ratio - w))
            top = int(random.uniform(0, h * ratio - h))

            expand_img[top:top + h, left:left + w] = img
            img = expand_img
            boxes += np.tile((left, top), 2)

            return img, boxes, labels


class RandomCrop(object):
    """Crop the image with bbox requirement.
    For image of W x H:
        existing bbox iou with cropped patch should >= min_ious.
        min_ious = min_ious + 1 + 0
        new_h = uniform of [min_crop_size*h, h]
        new_w = uniform of [min_crop_size*w, w]

    Args:
        min_ious (Iterable): ious.
        min_crop_size (float): ratio.

    Returns:
        img, boxes, labels.
    """
    def __init__(self, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3):
        # 1: return ori img
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size

    def __call__(self, img, boxes, labels):

        if boxes is None:
            h, w, c = img.shape
            while True:
                mode = random.choice(self.sample_mode)
                if mode == 1:
                    return img, boxes, labels

                for i in range(50):
                    new_w = random.uniform(self.min_crop_size * w, w)
                    new_h = random.uniform(self.min_crop_size * h, h)

                    # h / w in [0.5, 2]
                    if new_h / new_w < 0.5 or new_h / new_w > 2:
                        continue

                    return img, boxes, labels
        else:
            h, w, c = img.shape
            while True:
                mode = random.choice(self.sample_mode)
                if mode == 1:
                    return img, boxes, labels

                min_iou = mode
                for i in range(50):
                    new_w = random.uniform(self.min_crop_size * w, w)
                    new_h = random.uniform(self.min_crop_size * h, h)

                    # h / w in [0.5, 2]
                    if new_h / new_w < 0.5 or new_h / new_w > 2:
                        continue

                    left = random.uniform(w - new_w)
                    top = random.uniform(h - new_h)

                    patch = np.array((int(left), int(top), int(left + new_w),
                                      int(top + new_h)))
                    overlaps = bbox_overlaps_np(
                        patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                    if overlaps.min() < min_iou:
                        continue

                    # center of boxes should inside the crop img
                    center = (boxes[:, :2] + boxes[:, 2:]) / 2
                    mask = (center[:, 0] > patch[0]) * (
                            center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (
                                   center[:, 1] < patch[3])
                    if not mask.any():
                        continue
                    boxes = boxes[mask]
                    labels = labels[mask]

                    # adjust boxes
                    img = img[patch[1]:patch[3], patch[0]:patch[2]]
                    boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                    boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                    boxes -= np.tile(patch[:2], 2)

                    return img, boxes, labels


class Augmentation(object):
    """Extra augmentations except simple ones.

    Args:
        photo_metric_distortion: The loaded detector.
        expand: Either image files or loaded images.
        random_crop: Either image files or loaded images.

    Returns:
        img, boxes, labels.
    """
    def __init__(
            self,
            photo_metric_distortion=None,
            expand=None,
            random_crop=None
    ):
        self.transforms = []
        if photo_metric_distortion is not None:
            self.transforms.append(
                PhotoMetricDistortion(**photo_metric_distortion))
        if expand is not None:
            self.transforms.append(Expand(**expand))
        if random_crop is not None:
            self.transforms.append(RandomCrop(**random_crop))

    def __call__(self, img, boxes, labels):
        img = img.astype(np.float32)
        for transform in self.transforms:
            img, boxes, labels = transform(img, boxes, labels)

        return img, boxes, labels


def random_scale(img_scales, mode='range'):
    """Randomly select a scale from a list of scales or scale ranges.
    Ex.:
        img_scales = [(h1, w1)] -> (h1, w1)
        img_scales = [(h1, w1), (h2, w2)] -> value mode: (h2, w2)
                                          -> range mode: noob mode

    Args:
        img_scales (list[tuple]): Image scale or scale range.
        mode (str): "range" or "value".

    Returns:
        tuple: Sampled image scale.
    """
    num_scales = len(img_scales)
    if num_scales == 1:  # fixed scale is specified
        img_scale = img_scales[0]
    elif num_scales == 2:  # randomly sample a scale
        if mode == 'range':
            img_scale_long = [max(s) for s in img_scales]
            img_scale_short = [min(s) for s in img_scales]
            long_edge = np.random.randint(
                min(img_scale_long),
                max(img_scale_long) + 1)
            short_edge = np.random.randint(
                min(img_scale_short),
                max(img_scale_short) + 1)
            img_scale = (long_edge, short_edge)
        elif mode == 'value':
            img_scale = img_scales[np.random.randint(num_scales)]
    else:
        if mode != 'value':
            raise ValueError(
                'Only "value" mode supports more than 2 image scales')
        img_scale = img_scales[np.random.randint(num_scales)]

    return img_scale
