import torch
import mmcv
import numpy as np

from detectors.SSD_VGG16_Detector import SSDDetector
from utils.checkpoint import load_checkpoint
from cores.pre_processing.image_transform import ImageTransform
from mmcv.parallel.data_container import DataContainer
from utils.image_utils import to_tensor
from utils.image import imshow_det_bboxes

# checkpoints
CHECKPOINT_FILE = '/PATH/TO/CHECKPOINT'

# input
IMG_PATH = '/home/yuan/DLCariesScreen/deployment/example_1.jpg'

# output
SAVE_PATH = '/home/yuan/DLCariesScreen/deployment/example_1_result.jpg'

# image
NUM_CLASS = 2
IMG_SCALE = (300, 300)
flip = False
flip_ratio = 0
IMG_TRANSFORM_CONFIG = \
    dict(
        mean=[123.675, 116.28, 103.53],
        std=[128.0, 128.0, 128.0],
        to_rgb=True,
        pad_values=(0, 0, 0),
        resize_keep_ratio=True,
    )
THRESHOLD = [0.25939, 0.19710, 0.15921]

# loading
workers_per_gpu = 1
imgs_per_gpu = 1

# set True when input size does not vary a lot
torch.backends.cudnn.benchmark = True


def show_one_image(result, file_path, output_dir):

    found_image_bboxes = []
    found_image_bbox_labels = []

    for class_label, per_class_result in enumerate(result):
        # class_label: int
        # per_class_result: number-of-bboxes x [x, y, x, y, prob]
        for per_box_per_class_result in per_class_result:
            found_image_bbox_labels.append(class_label)
            found_image_bboxes.append(per_box_per_class_result)

    if len(found_image_bbox_labels) == 0:
        found_image_bboxes = np.zeros((0, 5))
        found_image_bbox_labels = np.zeros((0,))
    else:
        found_image_bboxes = np.asarray(found_image_bboxes).reshape((-1, 5))
        found_image_bbox_labels = np.asarray(found_image_bbox_labels).reshape((-1))

    img = mmcv.imread(file_path)
    height, width, _ = img.shape

    imshow_det_bboxes(
        img=img,
        bboxes=found_image_bboxes,
        labels=found_image_bbox_labels,
        score_thr=THRESHOLD,
        bbox_color=['green', ],
        show_uncertainty=False,
        thickness=np.int((height*width/480/480)**0.5),
        font_scale=np.float((height*width/480/480)**0.5)/2,
        show=False,
        win_name='',
        wait_time=0,
        out_file=SAVE_PATH,
    )


def main():

    # load image
    img = mmcv.imread(IMG_PATH)
    img_height = img.shape[0]
    img_width = img.shape[1]

    # image pre-processing
    img_transform = ImageTransform(
        mean=IMG_TRANSFORM_CONFIG['mean'],
        std=IMG_TRANSFORM_CONFIG['std'],
        to_rgb=IMG_TRANSFORM_CONFIG['to_rgb'],
    )
    img, img_shape, pad_shape, scale_factor = \
        img_transform(
            img=img, scale=IMG_SCALE, flip=False, pad_val=IMG_TRANSFORM_CONFIG['pad_values'], keep_ratio=IMG_TRANSFORM_CONFIG['resize_keep_ratio']
        )
    img_meta = dict(
        ori_shape=(img_height, img_width, 3),
        img_shape=img_shape,
        pad_shape=pad_shape,
        scale_factor=scale_factor,
        flip=False
    )
    data = dict(
        img=DataContainer(to_tensor(img), stack=True),
        img_meta=DataContainer(img_meta, cpu_only=True)
    )

    # define the model
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
        device='cpu',
    )

    # load checkpoint
    _ = load_checkpoint(
        model=model,
        filename=CHECKPOINT_FILE,
        map_location='cpu',
        strict=True,
        logger=None
    )

    # parallelize model
    model.eval()

    # results and progress bar
    # inference the data
    with torch.no_grad():
        result = model(is_test=True, rescale=True, img=data['img'].data.unsqueeze(0), img_meta=(data['img_meta'].data, ))

    show_one_image(result[0], IMG_PATH, SAVE_PATH)


if __name__ == '__main__':
    main()
