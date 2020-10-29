import torch.nn as nn
from mmcv.cnn import xavier_init, constant_init, kaiming_init, normal_init
from backbones.VGG16 import VGG


class SSDVGG(VGG):
    """
    SSDVGG(
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
          (30): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
          (31): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6))
          (32): ReLU(inplace=True)
          (33): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
          (34): ReLU(inplace=True)
        )
        (extra): Sequential(
          (0): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
          (1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (2): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
          (3): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
          (4): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
          (5): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
          (6): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
          (7): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
        )
        (l2_norm): L2Norm()
    """
    def __init__(
            self,
            extra_setting,
            out_feature_indices,
            out_extra_indices,
            use_dropout,
            dropout_rate,
    ):
        super(SSDVGG, self).__init__(
            with_bn=False,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
            num_classes=-1,
            num_stages=5,
            dilations=(1, 1, 1, 1, 1),
            out_indices=(3, 4),
            frozen_stages=-1,
            bn_eval=True,
            bn_frozen=False,
            ceil_mode=True,
            with_last_pool=False
        )
        self.extra_setting = extra_setting
        self.out_feature_indices = out_feature_indices
        self.out_extra_indices = out_extra_indices
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

        self.features.add_module(
            str(len(self.features)),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, dilation=1)
        )
        self.features.add_module(
            str(len(self.features)), nn.ReLU(inplace=True)
        )
        if use_dropout:
            self.features.add_module(
                str(len(self.features)),
                nn.Dropout(p=dropout_rate, inplace=False)
            )
        self.features.add_module(
            str(len(self.features)), nn.Conv2d(1024, 1024, kernel_size=1)
        )
        self.features.add_module(
            str(len(self.features)), nn.ReLU(inplace=True)
        )
        if use_dropout:
            self.features.add_module(
                str(len(self.features)),
                nn.Dropout(p=dropout_rate, inplace=False)
            )

        self.in_planes = 1024
        self.extra = self._make_extra_layers(self.extra_setting)

    def _make_extra_layers(self, out_planes):
        layers = []
        for i in range(len(out_planes)):
            conv = nn.Conv2d(self.in_planes, out_planes[i][1], kernel_size=out_planes[i][0], stride=out_planes[i][2], padding=out_planes[i][3])
            layers.append(conv)
            layers.append(nn.ReLU(inplace=True))
            if self.use_dropout:
                layers.append(
                    nn.Dropout(p=self.dropout_rate, inplace=False)
                )

            self.in_planes = out_planes[i][1]

        return nn.Sequential(*layers)

    def init_weights(self):
        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.01)

        for m in self.extra.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

        # constant_init(self.l2_norm, val=20., bias=0.)

    def forward(self, x):
        outs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_feature_indices:
                outs.append(x)
        for i, layer in enumerate(self.extra):
            x = layer(x)
            if i in self.out_extra_indices:
                outs.append(x)
        # outs[0] = self.l2_norm(outs[0])
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)
