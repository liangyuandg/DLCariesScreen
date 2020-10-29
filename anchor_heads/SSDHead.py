from __future__ import division

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

from cores.anchor.AnchorGenerator import AnchorGenerator
from cores.loss.smooth_l1_loss import smooth_l1_loss
from mmcv.cnn.weight_init import xavier_init
from cores.anchor.anchor_target_maxiou_pseudo import anchor_target
from cores.misc import multi_apply
from cores.bbox.transforms import delta2bbox
from cores.post_processing.nms import multiclass_nms


class SSDHead(nn.Module):
    """SSD-based head.
    Args:
        input_size ((int, int)): input size. (480, 320).
        num_classes (int): num_classes. 81.
        in_channels (Iterable): A list of number of channels of the feature map. (512, 1024, 512, 256, 256, 256).
        anchor_ratios (Iterable): Anchor aspect ratios. ([2], [2, 3], [2, 3], [2, 3], [2], [2]).
        anchor_strides (Iterable): Anchor strides. 1 pixel's real size on the feature map.  ((8, 8), (18, 19), (34, 36), (69, 64), (96, 107), (160, 320)).
        basesize_ratios (Iterable): Anchor base scales at different resolutions. Scales are [0, 1] scaled. (0.02, 0.06, 0.09, 0.12 0.16)
        target_means (Iterable): Mean values of regression targets. (.0, .0, .0, .0).
        target_stds (Iterable): Std values of regression targets. (1.0, 1.0, 1.0, 1.0).
        sampling (bool): true if do hard negative mining. FocalLoss will not need hard mining.
        neg_pos_ratio (int): negative samples / positive samples.
        smoothl1_beta (float): smooth L1 loss parameter.
        nms_pre (int): filter to get number of bboxes before nms.
        score_thr (float): score thr to be considered as positive.
        nms_cfg (List): list for configuring nms.
        max_per_img (int): max number of bbox to be considered in an image, after nms.
    """

    def __init__(
            self,
            # basic
            input_size,
            num_classes,
            in_channels,
            # anchor generate
            anchor_ratios,
            anchor_strides,
            basesizes,
            allowed_border,
            # regression
            target_means,
            target_stds,
            # box assign
            pos_iou_thr,
            neg_iou_thr,
            min_pos_iou,
            gt_max_assign_all,
            # sampling
            sampling,
            # balancing the loss
            neg_pos_ratio,
            # loss
            smoothl1_beta,
            # inference nms
            nms_pre,   # per level
            score_thr,
            min_size,
            max_scale_ratio,
            nms_cfg,
            max_per_img,
            device,
    ):
        super(SSDHead, self).__init__()

        # basic
        self.input_size = input_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        # anchor generate
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.basesizes = basesizes
        self.allowed_border = allowed_border
        # regression
        self.target_means = target_means
        self.target_stds = target_stds
        # sampling
        self.sampling = sampling
        self.neg_pos_ratio = neg_pos_ratio
        # loss
        self.smoothl1_beta = smoothl1_beta
        # box assign
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        # inference nms
        self.min_size = min_size
        self.max_scale_ratio = max_scale_ratio
        self.nms_pre = nms_pre
        self.score_thr = score_thr
        self.nms_cfg = nms_cfg
        self.max_per_img = max_per_img
        self.device = device

        """create final cls/bbox convolutions"""
        # [2, 3] -> [1/2, 1/3, 1, 2, 3] @ scale Sk + [1] @ scale sqrt(SkSk+1)
        num_anchors = [len(ratios) for ratios in anchor_ratios]
        reg_convs = []
        cls_convs = []
        for i in range(len(self.in_channels)):
            reg_convs.append(
                nn.Conv2d(
                    self.in_channels[i],
                    num_anchors[i] * 4,
                    kernel_size=3,
                    padding=1
                )
            )
            cls_convs.append(
                nn.Conv2d(
                    self.in_channels[i],
                    num_anchors[i] * num_classes,
                    kernel_size=3,
                    padding=1
                )
            )
        self.reg_convs = nn.ModuleList(reg_convs)
        self.cls_convs = nn.ModuleList(cls_convs)

        """generate anchors for each map"""
        self.anchor_generators = []

        for k in range(len(self.anchor_strides)):
            # base size: base_size_w, base_size_h
            base_size = self.basesizes[k]

            # left top bbox center
            stride_w, stride_h = anchor_strides[k]
            ctr = ((stride_w - 1) / 2., (stride_h - 1) / 2.)

            # scales
            scales = [1.0, ]

            # ratios
            ratios = self.anchor_ratios[k]

            anchor_generator = AnchorGenerator(
                base_size=base_size,
                scales=scales,
                ratios=ratios,
                scale_major=False,
                ctr=ctr
            )

            self.anchor_generators.append(anchor_generator)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)

    def forward(self, feats):
        """
        Args:
            feats: Number_of_levels x B x C x H x W.
        Returns:
            cls_scores: Number_of_levels x B x num_classes x H x W
            bbox_preds: Number_of_levels x B x num_anchors*4 x H x W
        """
        cls_scores = []
        bbox_preds = []
        for feat, reg_conv, cls_conv in zip(
                feats, self.reg_convs, self.cls_convs
        ):
            cls_scores.append(cls_conv(feat))
            bbox_preds.append(reg_conv(feat))

        return cls_scores, bbox_preds

    def get_anchors(self, featmap_sizes, img_metas):
        """

        *** This is for training! ***

        Get anchors according to feature map sizes.
        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes. [Number_of_levels, H, W]
            img_metas (list[dict]): Image meta info.

        Returns:
            anchor_list: [Number_of_images, Number_of_level, Anchor_per_location * N_of_locations, 4].
            valid_flag_list: [Number_of_images, Number_of_level, Anchor_per_location * N_of_locations].
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute anchors for one time
        # NOT TRUE FOR OTHER METHODS.
        multi_level_anchors = []
        for i in range(num_levels):
            # [Anchor_per_location * N_of_locations, 4]
            if self.device is None:
                anchors = self.anchor_generators[i].grid_anchors(
                    featmap_sizes[i], self.anchor_strides[i]
                )
            else:
                anchors = self.anchor_generators[i].grid_anchors(
                    featmap_sizes[i], self.anchor_strides[i], device='cpu'
                )
            # [Number_of_level, Anchor_per_location * N_of_locations, 4]
            multi_level_anchors.append(anchors)
        # [Number_of_images, Number_of_level, Anchor_per_location * N_of_locations, 4]
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags (for padding) of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / anchor_stride[1])), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride[0])), feat_w)
                if self.device is None:
                    flags = self.anchor_generators[i].valid_flags(
                        (feat_h, feat_w), (valid_feat_h, valid_feat_w)
                    )
                else:
                    flags = self.anchor_generators[i].valid_flags(
                        (feat_h, feat_w), (valid_feat_h, valid_feat_w), device='cpu'
                    )
                # [Number_of_level, Anchor_per_location * N_of_locations]
                multi_level_flags.append(flags)
            # [Number_of_images, Number_of_level, Anchor_per_location * N_of_locations]
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def get_bboxes(
            self,
            cls_scores,
            bbox_preds,
            img_metas,
            rescale=False
    ):
        """

        *** This is for inference! ***

        Args:
            cls_scores: Number_of_levels x Number_of_images x (num_anchors*num_classes) x H x W
            bbox_preds: Number_of_levels x Number_of_images x (num_anchors*4) x H x W
            img_metas: [Number_of_images, list].
            rescale (bool):
        Returns:
            result_list: [Number_of_images, {det_bboxes: k x 5, det_labels: k}].
        """

        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        if self.device is None:
            mlvl_anchors = [
                self.anchor_generators[i].grid_anchors(
                    featmap_size=cls_scores[i].size()[-2:],
                    stride=self.anchor_strides[i]
                )
                for i in range(num_levels)
            ]
        else:
            mlvl_anchors = [
                self.anchor_generators[i].grid_anchors(
                    featmap_size=cls_scores[i].size()[-2:],
                    stride=self.anchor_strides[i],
                    device='cpu'
                )
                for i in range(num_levels)
            ]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]

            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(
                cls_scores=cls_score_list,
                bbox_preds=bbox_pred_list,
                mlvl_anchors=mlvl_anchors,
                img_shape=img_shape,
                scale_factor=scale_factor,
                rescale=rescale
            )
            result_list.append(proposals)

        return result_list

    def _get_bboxes_single(
            self,
            cls_scores,
            bbox_preds,
            mlvl_anchors,
            img_shape,
            scale_factor,
            rescale,
    ):
        """
        Args:

            cls_scores: Number_of_levels x (num_anchors*num_classes) x H x W
            bbox_preds: Number_of_levels x (num_anchors*4) x H x W
            mlvl_anchors: Number_of_levels x Anchor_per_location*N_of_locations x 4
            img_shape: [H, W]
            scale_factor: int
            rescale (bool):.
        Returns:
            det_bboxes: k x 5
            det_labels: k
        """
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []

        for cls_score, bbox_pred, anchors in zip(cls_scores, bbox_preds, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.num_classes)

            scores = cls_score.softmax(-1)

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)

            if 0 < self.nms_pre < scores.shape[0]:
                max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(self.nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            bboxes = delta2bbox(anchors, bbox_pred, self.target_means, self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        # Number_of_levels x k x 4
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)

        # Number_of_levels x k
        mlvl_scores = torch.cat(mlvl_scores)

        det_bboxes, det_labels = multiclass_nms(
            multi_bboxes=mlvl_bboxes,
            multi_scores=mlvl_scores,
            score_thr=self.score_thr,
            min_size=self.min_size,
            max_scale_ratio=self.max_scale_ratio,
            nms_cfg=self.nms_cfg,
            max_num=self.max_per_img,
            score_factors=None
        )


        return det_bboxes, det_labels

    def _loss_single(
            self,
            cls_score,
            bbox_pred,
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            num_total_samples,
    ):
        """
        Args:
            cls_score: [Number_of_levels*H*W x num_classes].
            bbox_pred: [Number_of_levels*H*W*num_anchors x 4].
            labels: [N].
            label_weights: [N].
            bbox_targets: [Number_of_levels*Anchor_per_location*N_of_locations, 4].
            bbox_weights: [Number_of_levels*Anchor_per_location*N_of_locations, 4].
            num_total_samples: int for all images.
        Returns:
            loss_cls: B x num_classes x H x W
            loss_bbox: B x num_anchors x 4 x H x W
        """
        loss_cls_all = F.cross_entropy(
            cls_score, labels, reduction='none'
        ) * label_weights
        pos_inds = (labels > 0).nonzero().view(-1)
        neg_inds = (labels == 0).nonzero().view(-1)

        # balance pos/neg samples
        num_pos_samples = pos_inds.size(0)
        num_neg_samples = self.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)

        # calculate the loss after balancing
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples

        bbox_pred = torch.where(torch.isnan(bbox_pred), torch.zeros_like(bbox_pred), bbox_pred)
        bbox_pred = torch.where(torch.isinf(bbox_pred), torch.zeros_like(bbox_pred), bbox_pred)
        loss_bbox = smooth_l1_loss(
            pred=bbox_pred,
            target=bbox_targets,
            weight=bbox_weights,
            beta=self.smoothl1_beta,
            reduction='none',
            avg_factor=num_total_samples
        )

        return loss_cls[None], loss_bbox

    def loss(
            self,
            cls_scores,
            bbox_preds,
            gt_bboxes,
            gt_labels,
            img_metas,
    ):
        """
        Args:
            cls_scores: Number_of_levels x Number_of_images x (num_anchors*num_classes) x H x W
            bbox_preds: Number_of_levels x B x (num_anchors*4) x H x W
            gt_bboxes: [Number_of_images, k, 4].
            gt_labels: [Number_of_images, k].
            img_metas: [Number_of_images, list].
        Returns:
            loss_cls: [Number_of_images] of float.
            loss_bbox: [Number_of_images] of float.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        # anchor_list: [Number_of_images, Number_of_level, Anchor_per_location * N_of_locations, 4].
        # valid_flag_list: [Number_of_images, Number_of_level, Anchor_per_location * N_of_locations].
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas
        )

        cls_reg_targets = anchor_target(
            anchor_list=anchor_list,
            valid_flag_list=valid_flag_list,
            gt_bboxes_list=gt_bboxes,
            img_metas=img_metas,
            target_means=self.target_means,
            target_stds=self.target_stds,
            allowed_border=self.allowed_border,
            # maxiou assigner parameters
            pos_iou_thr=self.pos_iou_thr,
            neg_iou_thr=self.neg_iou_thr,
            min_pos_iou=self.min_pos_iou,
            gt_max_assign_all=self.gt_max_assign_all,
            gt_labels_list=gt_labels,
            # pseudo sampler parameters
            sampling=False
        )
        if cls_reg_targets is None:
            return None

        # labels_list:
        #   [Number_of_images, Number_of_levels, Anchor_per_location * N_of_locations]. label if pos. 0 if neg.
        # label_weights_list:
        #   [Number_of_images, Number_of_levels, Anchor_per_location * N_of_locations]. all 1s.
        # bbox_targets_list:
        #   [Number_of_images, Number_of_levels, Anchor_per_location * N_of_locations, 4]. deltas.
        # bbox_weights_list:
        #   [Number_of_images, Number_of_levels, Anchor_per_location * N_of_locations, 4]. 1 if pos. 0 if neg.
        # num_total_pos: int.
        # num_total_neg: int.
        labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg \
            = cls_reg_targets

        num_images = len(img_metas)

        # Number_of_levels x Number_of_images x num_classes x H x W
        #   -> Number_of_images x Number_of_levels*H*W x num_classes
        all_cls_scores = torch.cat(
            [s.permute(0, 2, 3, 1).reshape(num_images, -1, self.num_classes) for s in cls_scores],
            1
        )
        # Number_of_images x -1
        all_labels = torch.cat(labels_list, -1).view(num_images, -1)
        # Number_of_images x -1
        all_label_weights = torch.cat(label_weights_list, -1).view(num_images, -1)
        # Number_of_levels x Number_of_images x num_anchors*4 x H x W
        #   -> Number_of_images x Number_of_levels*H*W*num_anchors x 4
        all_bbox_preds = torch.cat(
            [b.permute(0, 2, 3, 1).reshape(num_images, -1, 4) for b in bbox_preds],
            -2
        )
        # [Number_of_images, Number_of_levels*Anchor_per_location*N_of_locations, 4].
        all_bbox_targets = torch.cat(bbox_targets_list, -2).view(num_images, -1, 4)
        # [Number_of_images, Number_of_levels*Anchor_per_location*N_of_locations, 4]
        all_bbox_weights = torch.cat(bbox_weights_list, -2).view(num_images, -1, 4)
        losses_cls, losses_bbox = multi_apply(
            self._loss_single,
            all_cls_scores,
            all_bbox_preds,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            num_total_samples=num_total_pos
        )

        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)




