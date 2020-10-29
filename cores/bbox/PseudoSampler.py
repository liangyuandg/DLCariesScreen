import torch
from cores.bbox.SamplingResult import SamplingResult


class PseudoSampler(object):
    """
    Args:
        num (int): num of gts.
        pos_fraction (float): desired positive sample ratio.
        neg_pos_ub (float): upper bound for neg/pos.
        add_gt_as_proposals (bool): if add gts in proposals.
    """
    def __init__(
            self,
            # num,
            # pos_fraction,
            # neg_pos_ub=-1,
            # add_gt_as_proposals=True,
    ):
        pass
        # self.num = num
        # self.pos_fraction = pos_fraction
        # self.neg_pos_ub = neg_pos_ub
        # self.add_gt_as_proposals = add_gt_as_proposals

    def sample(
            self,
            assign_result,
            bboxes,
            gt_bboxes
    ):
        """
        Args:
            num_gts: k.
            assigned_gt_inds: [N], -1, 0, or index of assigned gt (1-based).
            max_overlaps: [N], max of overlap with its nearest gt.
            assigned_labels: [N], label of the assigned gt. assigned_gt_inds of -1 and 0 will be ignored.
            bboxes: [n, 4].
            gt_bboxes: [k, 4].

        Returns:
            pos_inds: inds of positives in assigned_gt_inds.
            neg_inds: inds of zeros (negatives) in assigned_gt_inds.
            bboxes: [n, 4].
            gt_bboxes: [k, 4].
            assign_result: assign_result.
            gt_flags: [n], if they are gts.
        """

        pos_inds = torch.nonzero(
            assign_result.assigned_gt_inds > 0).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.assigned_gt_inds == 0).squeeze(-1).unique()
        gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=torch.uint8)
        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags)

        return sampling_result
