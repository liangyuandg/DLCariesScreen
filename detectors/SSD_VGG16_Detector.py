import torch.nn as nn

from backbones.SSD_VGG16 import SSDVGG
from anchor_heads.SSDHead import SSDHead
from cores.bbox.transforms import bbox2result


class SSDDetector(nn.Module):
    def __init__(
            self,
            # basic
            input_size,
            num_classes,
            in_channels,
            use_dropout,
            dropout_rate,
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
            nms_pre,
            score_thr,
            min_size,
            max_scale_ratio,
            nms_cfg,
            max_per_img,
            # device
            device,
    ):
        super(SSDDetector, self).__init__()
        # [kernel size, out_plane, stride, padding]
        self.backbone = SSDVGG(
            extra_setting=[[1, 256, 1, 0], [3, 512, 2, 1], [1, 128, 1, 0], [3, 256, 2, 1], [1, 128, 1, 0],
                           [3, 256, 2, 1]],
            out_feature_indices=(25, 31),
            out_extra_indices=(3, 7, 11, 15),
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
        )
        self.bbox_head = SSDHead(
            # basic
            input_size=input_size,
            num_classes=num_classes,
            in_channels=in_channels,
            # anchor generate
            anchor_ratios=anchor_ratios,
            anchor_strides=anchor_strides,
            basesizes=basesizes,
            allowed_border=allowed_border,
            # regression
            target_means=target_means,
            target_stds=target_stds,
            # box assign
            pos_iou_thr=pos_iou_thr,
            neg_iou_thr=neg_iou_thr,
            min_pos_iou=min_pos_iou,
            gt_max_assign_all=gt_max_assign_all,
            # sampling
            sampling=sampling,
            # balancing the loss
            neg_pos_ratio=neg_pos_ratio,
            # loss
            smoothl1_beta=smoothl1_beta,
            # inference nms
            nms_pre=nms_pre,
            score_thr=score_thr,
            min_size=min_size,
            max_scale_ratio=max_scale_ratio,
            nms_cfg=nms_cfg,
            max_per_img=max_per_img,
            device=device,
        )
        # init
        self.backbone.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        return x

    def forward(self, img, img_meta, is_test, **kwargs):
        if is_test:
            return self.forward_test(img, img_meta, **kwargs)
        else:
            return self.forward_train(img, img_meta, **kwargs)

    def forward_train(
            self,
            img,
            img_meta,
            gt_bboxes,
            gt_labels,
    ):
        """
        forward for training. return losses.
        """
        x = self.extract_feat(img)
        cls_scores, bbox_preds = self.bbox_head(x)
        losses = self.bbox_head.loss(
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            img_metas=img_meta,
        )

        return losses

    def forward_test(self, img, img_meta, rescale):

        # features: Number_of_levels x Number_of_images x features x H x W
        x = self.extract_feat(img)

        # cls_scores: Number_of_levels x Number_of_images x (num_anchors*num_classes) x H x W
        # bbox_preds: Number_of_levels x Number_of_images x (num_anchors*4) x H x W
        cls_score, bbox_pred = self.bbox_head(x)

        # [Number_of_images, tuple(det_bboxes: k x 4, det_labels: k)].
        bbox_list = self.bbox_head.get_bboxes(
            cls_scores=cls_score,
            bbox_preds=bbox_pred,
            img_metas=img_meta,
            rescale=rescale
        )

        # [Number_of_images, Number_of_classes, (reduced_k, 5)].
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]

        return bbox_results
