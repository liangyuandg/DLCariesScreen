import torch


class AssignResult(object):
    """
        num_gts: k.
        assigned_gt_inds: [N], -1, 0, or index of assigned gt (1-based).
        max_overlaps: [N], max of overlap with its nearest gt.
        assigned_labels: [N], label of the assigned gt. assigned_gt_inds of -1 and 0 will be ignored.
    """
    def __init__(self, num_gts, assigned_gt_inds, max_overlaps, assigned_labels=None):
        self.num_gts = num_gts
        self.assigned_gt_inds = assigned_gt_inds
        self.max_overlaps = max_overlaps
        self.assigned_labels = assigned_labels

    def add_gt_(self, gt_labels):
        self_inds = torch.arange(
            1, len(gt_labels) + 1, dtype=torch.long, device=gt_labels.device
        )
        self.assigned_gt_inds = torch.cat([self_inds, self.assigned_gt_inds])
        self.max_overlaps = torch.cat(
            [self.max_overlaps.new_ones(self.num_gts), self.max_overlaps])
        if self.labels is not None:
            self.assigned_labels = torch.cat([gt_labels, self.assigned_labels])
