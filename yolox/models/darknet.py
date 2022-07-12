#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from torch import nn

from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck
from .attention import CABlock, CBAM, NonLocalBlock, CrissCrossAttention

class Darknet(nn.Module):
    # number of blocks from dark2 to dark5.
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_out_channels=32,
        out_features=("dark3", "dark4", "dark5"),
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output channels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        self.stem = nn.Sequential(
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu"),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2),
        )
        in_channels = stem_out_channels * 2  # 64

        num_blocks = Darknet.depth2blocks[depth]
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[0], stride=2)
        )
        in_channels *= 2  # 128
        self.dark3 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[1], stride=2)
        )
        in_channels *= 2  # 256
        self.dark4 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )
        in_channels *= 2  # 512

        self.dark5 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
        )

    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu"),
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu",
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
            ]
        )
        return m

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class CSPDarknet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
            # CBAM(base_channels * 2, base_channels * 2)
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
            # CBAM(base_channels * 4, base_channels * 4)
            # CrissCrossAttention(base_channels * 4),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
            # CBAM(base_channels * 8, base_channels * 8),
            # CrissCrossAttention(base_channels * 8),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
            # CABlock(base_channels * 16, base_channels * 16),
            # CrissCrossAttention(base_channels * 16),
        )

        # self.one = CrissCrossAttention(base_channels * 4)
        # self.two = CrissCrossAttention(base_channels * 8)
        # self.three = CrissCrossAttention(base_channels * 16)


    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        # y1 = self.one(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        # y2 = self.two(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        # y3 = self.three(x)
        outputs["dark5"] = x
        visualize(x, )
        return {k: v for k, v in outputs.items() if k in self.out_features}



def visualize(self, masks, labels, img_paths, epoch, global_cam=None):
    select_id = [0, 1, 2, 3]
    maskt, maskm = masks[0], masks[1]
    h, w = maskt.shape[2], maskt.shape[3]
    maskt = maskt.data.cpu().numpy()
    test = maskt.data.cpu().numpy()
    maskm = maskm.data.cpu().numpy()

    for id in select_id:

        test[id] = maskt[id].view(H, W, C)
        result =
        for i in range(int(test[id][2])):
            masket[id] += test[id]**2

        mask = maskt[id].reshape(-1, h * w)
        mask = mask - np.min(mask, axis=1)
        mask = mask / np.max(mask, axis=1)
        # mask = 1 - mask
        mask = mask.reshape(h, w)
        # 归一化操作（最小的值为0，最大的为1）
        '''cam = weight[labels[id]:labels[id]+1].dot(global_cam[id].reshape((2048,h*w))).reshape(1,h * w)
        cam = cam - np.min(cam, axis=1)
        cam = cam / np.max(cam, axis=1)
        cam = cam.reshape(h, w)
        cam = np.uint8(255 * cam)'''
        # 转换为图片的255的数据
        cam_img = np.uint8(255 * mask)
        # resize 图片尺寸与输入图片一致
        # 将图片和CAM拼接在一起展示定位结果结果
        img = cv2.imread(img_paths[id], cv2.IMREAD_COLOR)
        img = cv2.resize(img, (128, 384), cv2.INTER_LINEAR)
        # img = cv2.cvtColor(np.uint8(imgs[0][i]), cv2.COLOR_RGB2BGR)
        heightv, widthv, _ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(cam_img, (widthv, heightv)), cv2.COLORMAP_JET)
        # heatmap_cam = cv2.applyColorMap(cv2.resize(cam, (widthv, heightv)), cv2.COLORMAP_JET)
        # 生成热度图
        result = heatmap * 0.3 + img * 0.5
        # result_cam = heatmap_cam * 0.3 + img * 0.5

        path = './heatmap/' + str(epoch)
        if not os.path.exists(path):
            os.mkdir(path)
        cv2.imwrite(path + '/' + img_paths[id].split('/')[-2] + str(labels[id].data) + '_t' + str(id) + '.jpg',
                    result)
