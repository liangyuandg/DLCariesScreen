import torch


class SamplingResult(object):

    def __init__(
            self,
            assigned_pos_inds,
            assigned_neg_inds,
            bboxes,
            gt_bboxes,
            assign_result,
            gt_flags
    ):
        """
        Args:
            assigned_pos_inds: inds of positives in assigned_gt_inds.
            assigned_neg_inds: inds of zeros (negatives) in assigned_gt_inds.
            bboxes: [n, 4].
            gt_bboxes: [k, 4].
            assign_result: assign_result.
                num_gts: k.
                assigned_gt_inds: [N], -1, 0, or index of assigned gt (1-based).
                max_overlaps: [N], max of overlap with its nearest gt.
                assigned_labels: [N], label of the assigned gt. assigned_gt_inds of -1 and 0 will be ignored.
            gt_flags: [n], if they are gts.
        Returns:
            assigned_pos_bboxes: [x, 4]. assigned positive bboxs.
            assigned_neg_bboxes: [y, 4]. assigned negative bboxs.
            pos_is_gt: [x]. if this case is from a gt. 0/1.
            num_gts: k.
            pos_assigned_gt_inds: [n]. indices of gts for the positive assigns.
            pos_gt_bboxes: [x, 4]. assigned gt boxes.
            pos_gt_labels: [x]. assigned bbox labels.

        """
        self.assigned_pos_inds = assigned_pos_inds
        self.assigned_neg_inds = assigned_neg_inds
        self.assigned_pos_bboxes = bboxes[assigned_pos_inds]
        self.assigned_neg_bboxes = bboxes[assigned_neg_inds]
        self.pos_is_gt = gt_flags[assigned_pos_inds]

        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.assigned_gt_inds[assigned_pos_inds] - 1
        self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds, :]
        if assign_result.assigned_labels is not None:
            self.pos_gt_labels = assign_result.assigned_labels[assigned_pos_inds]
        else:
            self.pos_gt_labels = None

    @property
    def bboxes(self):
        return torch.cat([self.assigned_pos_bboxes, self.assigned_neg_bboxes])
