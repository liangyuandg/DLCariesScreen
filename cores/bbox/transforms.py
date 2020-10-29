import numpy as np
import torch


def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5) or (n, 6)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class. num_classes x k x 5
    """
    if bboxes.shape[0] == 0:
        if bboxes.shape[1] == 5:
            return [
                np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)
            ]
        if bboxes.shape[1] == 6:
            return [
                np.zeros((0, 6), dtype=np.float32) for i in range(num_classes - 1)
            ]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes - 1)]


def bbox2delta(proposals, gt, means=[0, 0, 0, 0], stds=[1, 1, 1, 1]):
    """Convert proposed bbox to deltas.
    All values use pixels as units.

    regression result (deltas): tx, ty, tw, th.

    predicted bbox: xb, yb, wb, hb.
    anchor bbox: xa, ya, wa, ha.
    delta(x) = (xb - xa) / wa
    delta(y) = (yb - ya) / ha
    delta(w) = log(wb / wa)
    delta(h) = log(hb / ha)

    xb = (xa + delta(x) * wa)
    yb = (ya + delta(y) * ha)
    wb = e^delta(w) * wa
    hb = e^delta(h) * ha

    Args:
        proposals: N * [la=xa-0.5wa+0.5, ta=ya-0.5ha+0.5, ra=xa+0.5wa-0.5, ba=ya+0.5ha-0.5].
        gt: N * [lb_gt=xb_gt-0.5wb_gt+0.5, tb_gt=yb_gt-0.5hb_gt+0.5, rb_gt=xb_gt+0.5wb_gt-0.5, bb_gt=yb_gt+0.5hb_gt-0.5].
        means: means.
        stds: stds.

    Returns:
        list(ndarray): bbox results of each class
    """
    assert proposals.size() == gt.size()

    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5  # xa
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5  # ya
    pw = proposals[..., 2] - proposals[..., 0] + 1.0   # wa
    ph = proposals[..., 3] - proposals[..., 1] + 1.0   # ha

    gx = (gt[..., 0] + gt[..., 2]) * 0.5    # xb_gt
    gy = (gt[..., 1] + gt[..., 3]) * 0.5    # yb_gt
    gw = gt[..., 2] - gt[..., 0] + 1.0     # wb_gt
    gh = gt[..., 3] - gt[..., 1] + 1.0     # hb_gt

    dx = (gx - px) / pw     # (xb_gt-xa)/wa
    dy = (gy - py) / ph     # (yb_gt-ya)/ha
    dw = torch.log(gw / pw) # (wb_gt+1)/wa
    dh = torch.log(gh / ph) # (hb_gt+1)/ha
    deltas = torch.stack([dx, dy, dw, dh], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta2bbox(
        rois,
        deltas,
        means=[0, 0, 0, 0],
        stds=[1, 1, 1, 1],
        max_shape=None,
        wh_ratio_clip=16/1000
):
    """Convert deltas to bbox.
    regression result (deltas): tx, ty, tw, th.

    Args:
        rois: anchors. Anchor_per_location*N_of_locations x 4 (minx, miny, maxx, maxy)
        deltas: num_anchors*H*W x 4 (tx, ty, tw, th)
        means: means.
        stds: stds.
        max_shape: [int, int]
        wh_ratio_clip: float. for w/wa and h/ha.

    Returns:
        list(ndarray): k x 4 (minx, miny, maxx, maxy)

    predicted bbox: xb, yb, wb, hb.
    anchor bbox: xa, ya, wa, ha.
    delta(x) = (xb - xa) / wa
    delta(y) = (yb - ya) / ha
    delta(w) = log(wb / wa)
    delta(h) = log(hb / ha)

    xb = (xa + delta(x) * wa)
    yb = (ya + delta(y) * ha)
    wb = e^delta(w) * wa
    hb = e^delta(h) * ha
    """
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 4)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 4)
    denorm_deltas = deltas * stds + means

    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]

    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)

    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)   # xa
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)   # ya
    pw = (rois[:, 2] - rois[:, 0] + 1.0).unsqueeze(1).expand_as(dw) # wa
    ph = (rois[:, 3] - rois[:, 1] + 1.0).unsqueeze(1).expand_as(dh) # ha

    gw = pw * dw.exp()
    gh = ph * dh.exp()
    gx = torch.addcmul(px, 1, pw, dx)  # gx = px + pw * dx
    gy = torch.addcmul(py, 1, ph, dy)  # gy = py + ph * dy

    x1 = gx - gw * 0.5 + 0.5
    y1 = gy - gh * 0.5 + 0.5
    x2 = gx + gw * 0.5 - 0.5
    y2 = gy + gh * 0.5 - 0.5

    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)

    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view_as(deltas)

    return bboxes
