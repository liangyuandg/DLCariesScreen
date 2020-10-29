import torch

from cores.misc import multi_apply
from cores.bbox.MaxIoUAssigner import MaxIoUAssigner
from cores.bbox.PseudoSampler import PseudoSampler
from cores.bbox.transforms import bbox2delta


def anchor_target(
        anchor_list,
        valid_flag_list,
        gt_bboxes_list,
        img_metas,
        target_means,
        target_stds,
        allowed_border,
        # maxiou assigner parameters
        pos_iou_thr,
        neg_iou_thr,
        min_pos_iou,
        gt_max_assign_all,
        gt_labels_list=None,
        # pseudo sampler parameters
        sampling=False,
):

    """Compute regression and classification targets for anchors.

    Args:
        anchor_list (list[list]): Multi level anchors of each image.
            [Number_of_images, Number_of_levels, Anchor_per_location * N_of_locations, 4]
        valid_flag_list (list[list]): Multi level valid flags of each image.
            [Number_of_images, Number_of_levels, Anchor_per_location * N_of_locations]
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            [Number_of_images, k, 4]
        img_metas (list[dict]): Meta info of each image.
        target_means (Iterable): Mean value of regression targets. [float, float, float, float].
        target_stds (Iterable): Std value of regression targets. [float, float, float, float].
        allowed_border: int.

        pos_iou_thr (float): .
        neg_iou_thr (float): .
        min_pos_iou (float): .
        gt_max_assign_all (bool): .
        gt_labels_list: [Number_of_images, k].

    Returns:
        labels_list: [Number_of_images, Number_of_levels, Anchor_per_location * N_of_locations]. label if pos. 0 if neg.
        label_weights_list: [Number_of_images, Number_of_levels, Anchor_per_location * N_of_locations]. all 1s.
        bbox_targets_list: [Number_of_images, Number_of_levels, Anchor_per_location * N_of_locations, 4]. deltas.
        bbox_weights_list: [Number_of_images, Number_of_levels, Anchor_per_location * N_of_locations, 4]. 1 if pos. 0 if neg.
        num_total_pos: int.
        num_total_neg: int.
    """

    num_imgs = len(img_metas)
    assert len(anchor_list) == len(valid_flag_list) == num_imgs

    # anchor number of multi levels
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    # concat all level anchors and flags to a single tensor
    # anchor_list: [Number_of_images, Numner_of_levels*Anchor_per_location*N_of_locations, 4]
    # valid_flag_list: [Number_of_images, Numner_of_levels*Anchor_per_location*N_of_locations]
    for i in range(num_imgs):
        assert len(anchor_list[i]) == len(valid_flag_list[i])
        anchor_list[i] = torch.cat(anchor_list[i])
        valid_flag_list[i] = torch.cat(valid_flag_list[i])

    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]

    # do for each image
    (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights, pos_inds_list, neg_inds_list) \
        = multi_apply(
            anchor_target_single,
            anchor_list,
            valid_flag_list,
            gt_bboxes_list,
            gt_labels_list,
            img_metas,
            allowed_border=allowed_border,
            target_means=target_means,
            target_stds=target_stds,
            pos_iou_thr=pos_iou_thr,
            neg_iou_thr=neg_iou_thr,
            min_pos_iou=min_pos_iou,
            gt_max_assign_all=gt_max_assign_all,
    )

    # no valid anchors
    if any([labels is None for labels in all_labels]):
        return None

    # sampled anchors of all images
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])

    # split targets to a list w.r.t. multiple levels
    labels_list = _images_to_levels(all_labels, num_level_anchors)
    label_weights_list = _images_to_levels(all_label_weights, num_level_anchors)
    bbox_targets_list = _images_to_levels(all_bbox_targets, num_level_anchors)
    bbox_weights_list = _images_to_levels(all_bbox_weights, num_level_anchors)

    return labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg


def _images_to_levels(target, num_level_anchors):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end].squeeze(0))
        start = end
    return level_targets


def anchor_target_single(
        flat_anchors,
        valid_flags,
        gt_bboxes,
        gt_labels,
        img_meta,
        allowed_border,
        target_means,
        target_stds,
        pos_iou_thr,
        neg_iou_thr,
        min_pos_iou,
        gt_max_assign_all,
):
    """
    Args:
        flat_anchors: [Number_of_levels*Anchor_per_location * N_of_locations, 4] = [n, 4]
        valid_flags: [Number_of_levels*Anchor_per_location * N_of_locations] = [n]
        gt_bboxes: Ground truth bboxes of each image. [k, 4]
        img_metas: Meta info of each image.
        allowed_border: int for boarder to be allowed.
        target_means: Mean value of regression targets.
        target_stds: Std value of regression targets.
        pos_iou_thr: >thr is pos.
        neg_iou_thr: <thr is -1.
        min_pos_iou: <thr is 0.
        gt_max_assign_all: gt box labels all bboxs of the same iou.

    Returns:
        tuple
    """
    # [n]. if bbox is valid.
    inside_flags = anchor_inside_flags(
        flat_anchors,
        valid_flags,
        img_meta['img_shape'][:2],
        allowed_border=allowed_border,
    )
    if not inside_flags.any():
        return (None, ) * 6
    # valid anchors only
    anchors = flat_anchors[inside_flags.bool(), :]

    # assign bboxs to gts, and sampling.
    # assigned_pos_bboxes: [x, 4]. assigned positive bboxs.
    # assigned_neg_bboxes: [y, 4]. assigned negative bboxs.
    # pos_is_gt: [x]. if this case is from a gt.0 / 1.
    # num_gts: k.
    # pos_assigned_gt_inds: [n].indices of gts for the positive assigns.
    # pos_gt_bboxes: [x, 4]. assigned gt boxes.
    # pos_gt_labels: [x]. assigned bbox labels.
    bbox_assigner = MaxIoUAssigner(
        pos_iou_thr=pos_iou_thr,
        neg_iou_thr=neg_iou_thr,
        min_pos_iou=min_pos_iou,
        gt_max_assign_all=gt_max_assign_all,
    )

    # print('debug:')
    # print('len(gt_bboxes)')
    # print(len(gt_bboxes))
    # print('gt_bboxes[0].shape')
    # print(gt_bboxes[0].shape)
    assign_result \
        = bbox_assigner.assign(bboxes=anchors, gt_bboxes=gt_bboxes, gt_labels=gt_labels)

    bbox_sampler = PseudoSampler()
    sampling_result = bbox_sampler.sample(
        assign_result=assign_result, bboxes=anchors, gt_bboxes=gt_bboxes
    )

    # num_valid_anchors: int
    # bbox_targets: N * [delta(x), delta(y), delta(w), delta(h)]
    # bbox_weights: N * [int, int, int, int]
    # labels: N * [int]
    # label_weights: N * [float]
    num_valid_anchors = anchors.shape[0]

    bbox_targets = torch.zeros_like(anchors)
    bbox_weights = torch.zeros_like(anchors)
    labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
    label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

    pos_inds = sampling_result.assigned_pos_inds
    neg_inds = sampling_result.assigned_neg_inds
    if len(pos_inds) > 0:
        pos_bbox_targets = bbox2delta(
            sampling_result.assigned_pos_bboxes,
            sampling_result.pos_gt_bboxes,
            target_means,
            target_stds
        )
        bbox_targets[pos_inds, :] = pos_bbox_targets
        bbox_weights[pos_inds, :] = 1.0

        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]

        label_weights[pos_inds] = 1.0

    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0

    # map up to original set of anchors
    num_total_anchors = flat_anchors.size(0)
    labels = unmap(labels, num_total_anchors, inside_flags)
    label_weights = unmap(label_weights, num_total_anchors, inside_flags)
    bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
    bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

    return labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds


def anchor_inside_flags(
        flat_anchors,
        valid_flags,
        img_shape,
        allowed_border=0
):
    """return valid flags for anchors that are within the boarder requirements.

    Args:
        flat_anchors (list[list]): [Number_of_levels*Anchor_per_location * N_of_locations, 4].
        valid_flags (list[list]): [Number_of_levels*Anchor_per_location * N_of_locations].
        img_shape (list[int]): [H, W]
        allowed_border (int): boarder length outside the feature map.

    Returns:
        updated valid_flags (list[list]): [Number_of_levels*Anchor_per_location * N_of_locations].
    """
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & \
                       (flat_anchors[:, 0] >= -allowed_border) & \
                       (flat_anchors[:, 1] >= -allowed_border) & \
                       (flat_anchors[:, 2] < img_w + allowed_border) & \
                       (flat_anchors[:, 3] < img_h + allowed_border)
    else:
        inside_flags = valid_flags

    return inside_flags


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds.bool()] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds.bool(), :] = data
    return ret
