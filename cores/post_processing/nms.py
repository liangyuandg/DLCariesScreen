import torch
from torchvision.ops import nms


def multiclass_nms(
        multi_bboxes,
        multi_scores,
        score_thr,
        min_size,
        max_scale_ratio,
        nms_cfg,
        max_num=-1,
        score_factors=None
):
    """NMS for multi-class bboxes.
    PERFORM NMS CLASS-INDEPENDENTLY.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class)
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_cfg (List): nms configurations.
            [nms, dets, iou_thr, device_id]
            [soft_nms, dets, iou_thr, method, sigma, min_score]
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    # background included.
    filtered_multi_bboxes, filtered_multi_scores = [], []
    for per_box, per_score in zip(multi_bboxes, multi_scores):
        if min_size is not None and (per_box[3]-per_box[1]+1) * (per_box[2]-per_box[0]+1) < min_size:
            continue
        if max_scale_ratio is not None and ((per_box[3]-per_box[1]+1) / (per_box[2]-per_box[0]+1) > max_scale_ratio or (per_box[2]-per_box[0]+1) / (per_box[3]-per_box[1]+1) > max_scale_ratio):
            continue
        filtered_multi_bboxes.append(per_box)
        filtered_multi_scores.append(per_score)
    filtered_multi_bboxes = torch.stack(filtered_multi_bboxes, 0)
    filtered_multi_scores = torch.stack(filtered_multi_scores, 0)

    num_classes = filtered_multi_scores.shape[1]
    bboxes, labels = [], []
    for i in range(1, num_classes):
        cls_inds = filtered_multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if filtered_multi_bboxes.shape[1] == 4:
            _bboxes = filtered_multi_bboxes[cls_inds, :]
        else:
            _bboxes = filtered_multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
        _scores = filtered_multi_scores[cls_inds, i]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)

        cls_dets_indices = nms(boxes=cls_dets[:, 0:4], scores=cls_dets[:, 4], iou_threshold=nms_cfg[1])

        cls_labels = filtered_multi_bboxes.new_full(
            (cls_dets_indices.shape[0], ), i - 1, dtype=torch.long)
        bboxes.append(cls_dets[cls_dets_indices, :])
        labels.append(cls_labels)
    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:
            _, inds = bboxes[:, -1].sort(descending=True)
            inds = inds[:max_num]
            bboxes = bboxes[inds]
            labels = labels[inds]
    else:
        bboxes = filtered_multi_bboxes.new_zeros((0, 5))
        labels = filtered_multi_bboxes.new_zeros((0, ), dtype=torch.long)

    return bboxes, labels


def box_multiclass_nms(
        multi_bboxes,
        multi_scores,
        score_thr,
        nms_cfg,
        score_factors=None,
        min_size=None,
        max_scale_ratio=None,
):
    """NMS for multi-class bboxes.
    PERFORM NMS CLASS-INDEPENDENTLY.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class)
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_cfg (List): nms configurations.
            [nms, dets, iou_thr, device_id]
            [soft_nms, dets, iou_thr, method, sigma, min_score]
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS
        min_size (int):.
        max_scale_ratio(float):.

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    # reduce wired boxes (very thin, very high, very small)
    filtered_multi_bboxes, filtered_multi_scores = [], []
    for per_box, per_score in zip(multi_bboxes, multi_scores):
        if min_size is not None and (per_box[3]-per_box[1]+1) * (per_box[2]-per_box[0]+1) < min_size:
            continue
        if max_scale_ratio is not None and ((per_box[3]-per_box[1]+1) / (per_box[2]-per_box[0]+1) > max_scale_ratio or (per_box[2]-per_box[0]+1) / (per_box[3]-per_box[1]+1) > max_scale_ratio):
            continue
        filtered_multi_bboxes.append(per_box)
        filtered_multi_scores.append(per_score)
    filtered_multi_bboxes = torch.stack(filtered_multi_bboxes, 0)
    filtered_multi_scores = torch.stack(filtered_multi_scores, 0)

    # background not included.
    num_classes = multi_scores.shape[1]
    bboxes, labels = [], []
    for i in range(0, num_classes):
        cls_inds = filtered_multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if filtered_multi_bboxes.shape[1] == 4:
            _bboxes = filtered_multi_bboxes[cls_inds, :]
        else:
            _bboxes = filtered_multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
        _scores = filtered_multi_scores[cls_inds, i]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        cls_dets = torch.cat([_bboxes, _scores[:, None]], dim=1)

        # nms
        cls_dets_indices = nms(boxes=cls_dets[:, 0:4], scores=cls_dets[:, 4], iou_threshold=nms_cfg[2])

        cls_labels = filtered_multi_bboxes.new_full(
            (cls_dets_indices.shape[0], ), i, dtype=torch.long)
        bboxes.append(cls_dets[cls_dets_indices, :])
        labels.append(cls_labels)

    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
    else:
        bboxes = bboxes.new_zeros((0, 5))
        labels = bboxes.new_zeros((0, ), dtype=torch.long)

    return bboxes, labels


def box_multiclass_nms_w_dropout(
        multi_bboxes,
        multi_scores,
        multi_score_vars,
        score_thr,
        nms_cfg,
        score_factors=None,
        min_size=None,
        max_scale_ratio=None,
):
    """NMS for multi-class bboxes.
    PERFORM NMS CLASS-INDEPENDENTLY.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class)
        multi_score_vars (Tensor): shape (n, #class)
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_cfg (List): nms configurations.
            [nms, dets, iou_thr, device_id]
            [soft_nms, dets, iou_thr, method, sigma, min_score]
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS
        min_size (int):.
        max_scale_ratio(float):.

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    # reduce wired boxes (very thin, very high, very small)
    filtered_multi_bboxes, filtered_multi_scores, filtered_multi_score_vars = [], [], []
    for per_box, per_score, per_score_var in zip(multi_bboxes, multi_scores, multi_score_vars):
        if min_size is not None and (per_box[3]-per_box[1]+1) * (per_box[2]-per_box[0]+1) < min_size:
            continue
        if max_scale_ratio is not None and ((per_box[3]-per_box[1]+1) / (per_box[2]-per_box[0]+1) > max_scale_ratio or (per_box[2]-per_box[0]+1) / (per_box[3]-per_box[1]+1) > max_scale_ratio):
            continue
        filtered_multi_bboxes.append(per_box)
        filtered_multi_scores.append(per_score)
        filtered_multi_score_vars.append(per_score_var)
    filtered_multi_bboxes = torch.stack(filtered_multi_bboxes, 0)
    filtered_multi_scores = torch.stack(filtered_multi_scores, 0)
    filtered_multi_score_vars = torch.stack(filtered_multi_score_vars, 0)

    # background not included.
    num_classes = multi_scores.shape[1]
    bboxes, labels = [], []
    for i in range(0, num_classes):
        cls_inds = filtered_multi_scores[:, i] > score_thr
        if not cls_inds.any():
            continue
        # get bboxes and scores of this class
        if filtered_multi_bboxes.shape[1] == 4:
            _bboxes = filtered_multi_bboxes[cls_inds, :]
        else:
            _bboxes = filtered_multi_bboxes[cls_inds, i * 4:(i + 1) * 4]
        _scores = filtered_multi_scores[cls_inds, i]
        _score_vars = filtered_multi_score_vars[cls_inds, i]
        if score_factors is not None:
            _scores *= score_factors[cls_inds]
        cls_dets = torch.cat([_bboxes, _scores[:, None], _score_vars[:, None]], dim=1)

        # nms
        cls_dets_ind = nms(boxes=cls_dets[:, 0:4], scores=cls_dets[:, 4], iou_threshold=nms_cfg[2])

        cls_labels = filtered_multi_bboxes.new_full(
            (cls_dets_ind.shape[0], ), i, dtype=torch.long)
        bboxes.append(cls_dets[cls_dets_ind, :])
        labels.append(cls_labels)

    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
    else:
        bboxes = bboxes.new_zeros((0, 6))
        labels = bboxes.new_zeros((0, ), dtype=torch.long)

    return bboxes, labels
