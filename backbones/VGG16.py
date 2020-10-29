import torch.nn as nn
from mmcv.cnn.weight_init import constant_init, normal_init, kaiming_init
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, dilation):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        padding=dilation,
        dilation=dilation
    )


def make_vgg_layer(
        in_planes,
        out_planes,
        num_blocks,
        dilation,
        with_bn,
        ceil_mode,
        use_dropout,
        dropout_rate,
):
    layers = []
    for _ in range(num_blocks):
        layers.append(conv3x3(in_planes, out_planes, dilation))
        if with_bn:
            layers.append(nn.BatchNorm2d(out_planes))
        layers.append(nn.ReLU(inplace=True))
        if use_dropout:
            layers.append(nn.Dropout(p=dropout_rate, inplace=False))
        in_planes = out_planes
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode))

    return layers


class VGG(nn.Module):
    """VGG backbone.

    ~134,000,000 params

    (features): Sequential(
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU(inplace=True)
      (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (3): ReLU(inplace=True)
      (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
      (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (6): ReLU(inplace=True)
      (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (8): ReLU(inplace=True)
      (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
      (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (11): ReLU(inplace=True)
      (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (13): ReLU(inplace=True)
      (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (15): ReLU(inplace=True)
      (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
      (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (18): ReLU(inplace=True)
      (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (20): ReLU(inplace=True)
      (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (22): ReLU(inplace=True)
      (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
      (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (25): ReLU(inplace=True)
      (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (27): ReLU(inplace=True)
      (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (29): ReLU(inplace=True)
    )

    Args:
        with_bn (bool): Use BatchNorm or not.
        num_classes (int): number of classes for classification. -1 for non FC layers.
        num_stages (int): VGG stages, normally 5.
        dilations (Sequence[int]): Dilation of each stage. (1, 1, 1, 1, 1) for no dilation.
        out_indices (Sequence[int]): Output from which stages. (0, 1, 2, 3, 4) for every stage.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 for
            not freezing any parameters.
        bn_eval (bool): Whether to set BN layers as eval mode, namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of BN layers.
        ceil_mode (bool): how to resolve downsampling %2 issue.
        with_last_pool (bool): whether to pool the last conv layer.
    """

    # arch_settings = [2, 2, 3, 3, 3, 3]
    # plane_settings = [64, 128, 256, 512, 512, 128]
    arch_settings = [2, 2, 3, 3, 3]
    plane_settings = [64, 128, 256, 512, 512]

    def __init__(
            self,
            with_bn,
            num_classes,
            num_stages,
            dilations,
            out_indices,
            frozen_stages,
            bn_eval,
            bn_frozen,
            ceil_mode,
            with_last_pool,
            use_dropout,
            dropout_rate,
    ):
        super(VGG, self).__init__()

        assert len(dilations) == num_stages

        self.num_classes = num_classes
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.in_planes = 3
        self.range_sub_modules = []
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

        start_idx = 0
        vgg_layers = []
        for i, num_blocks in enumerate(self.arch_settings):
            num_modules = num_blocks * (2 + with_bn) + 1
            end_idx = start_idx + num_modules
            dilation = dilations[i]
            out_planes = self.plane_settings[i]
            vgg_layer = make_vgg_layer(
                in_planes=self.in_planes,
                out_planes=out_planes,
                num_blocks=num_blocks,
                dilation=dilation,
                with_bn=with_bn,
                ceil_mode=ceil_mode,
                use_dropout=self.use_dropout,
                dropout_rate=self.dropout_rate,
            )
            vgg_layers.extend(vgg_layer)
            self.in_planes = out_planes
            self.range_sub_modules.append([start_idx, end_idx])
            start_idx = end_idx

        if not with_last_pool:
            vgg_layers.pop(-1)
        self.add_module('features', nn.Sequential(*vgg_layers))

        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(128 * 5 * 5, 1024),
                nn.ReLU(True),
                # nn.Dropout(),
                nn.Linear(1024, 256),
                nn.ReLU(True),
                # nn.Dropout(),
                nn.Linear(256, num_classes),
            )

        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.01)

    # def forward(self, x):
    #     outs = []
    #     for i, layer in enumerate(self.features):
    #         x = layer(x)
    #         if i in self.out_indices:
    #             outs.append(x)
    #     if self.num_classes > 0:
    #         x = x.view(x.size(0), -1)
    #         x = self.classifier(x)
    #         outs.append(x)
    #     if len(outs) == 1:
    #         return outs[0]
    #     else:
    #         return tuple(outs)

    def forward(self, img, gt_labels, is_test):
        if is_test:
            return self.forward_test(img)
        else:
            return self.forward_train(img, gt_labels)

    def forward_train(
            self,
            x,
            gt_labels,
    ):
        """
        forward for training. return losses.
        """
        for i, layer in enumerate(self.features):
            x = layer(x)
        if self.num_classes > 0:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)

        losses = self.loss(
            cls_scores=x,
            gt_labels=gt_labels,
        )

        return losses

    def forward_test(self, x):
        for i, layer in enumerate(self.features):
            x = layer(x)
        if self.num_classes > 0:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        x = F.sigmoid(x).cpu().numpy()

        return x

    def loss(
            self,
            cls_scores,
            gt_labels,
    ):
        """
        Args:
            cls_scores: Number_of_images x num_classes
            gt_labels: [Number_of_images, num_classes].
        Returns:
            loss_cls: [Number_of_images] of float.
            loss_bbox: [Number_of_images] of float.
        """
        loss_cls_all = F.binary_cross_entropy_with_logits(
            input=cls_scores,
            target=gt_labels,
            weight=None,
            reduction='none',
        )

        return dict(loss_cls=loss_cls_all)



