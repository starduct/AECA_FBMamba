import torch
import torch.nn as nn
import torch.nn.functional as F
from . import DA_Block


class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class RPBlock(nn.Module):
    def __init__(self, input_chs, ratios=[1, 0.5, 0.25], bn_momentum=0.1):
        super(RPBlock, self).__init__()
        self.branches = nn.ModuleList()
        for i, ratio in enumerate(ratios):
            conv = nn.Sequential(
                nn.Conv2d(input_chs, int(input_chs * ratio),
                          kernel_size=(2 * i + 1), stride=1, padding=i),
                nn.BatchNorm2d(int(input_chs * ratio), momentum=bn_momentum),
                nn.ReLU()
            )
            self.branches.append(conv)

        self.fuse_conv = nn.Sequential(  # + input_chs // 64
            nn.Conv2d(int(input_chs * sum(ratios)), input_chs,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(input_chs, momentum=bn_momentum),
            nn.ReLU()
        )

    def forward(self, x):
        branches = torch.cat([branch(x) for branch in self.branches], dim=1)
        output = self.fuse_conv(branches) + x
        return output


class MSCA_SIM(nn.Module):
    # 去掉了这里的shrotcut，这里的3x3卷积，去掉了groups和bias

    def __init__(self, inp, oup, bn_momentum=0.1):
        super(MSCA_SIM, self).__init__()
        # input
        self.conv33 = nn.Conv2d(inp, oup, kernel_size=3, padding=1, bias=False)
        # split into multipats of multiscale attention
        self.conv17_0 = nn.Conv2d(inp, inp, (1, 7), padding=(0, 3), groups=inp)
        self.conv17_1 = nn.Conv2d(inp, oup, (7, 1), padding=(3, 0), groups=oup)
        # self.conv19_0 = nn.Conv2d(inp, inp, (1, 9), padding=(0, 4), groups=inp)
        # self.conv19_1 = nn.Conv2d(inp, oup, (9, 1), padding=(4, 0), groups=oup)
        self.conv15_0 = nn.Conv2d(inp, inp, (1, 5), padding=(0, 2), groups=inp)
        self.conv15_1 = nn.Conv2d(inp, oup, (5, 1), padding=(2, 0), groups=oup)

        self.bn1 = nn.BatchNorm2d(inp, momentum=bn_momentum)
        self.bn2 = nn.BatchNorm2d(inp, momentum=bn_momentum)
        self.bn3 = nn.BatchNorm2d(inp, momentum=bn_momentum)
        self.bn4 = nn.BatchNorm2d(inp, momentum=bn_momentum)
        self.bn33 = nn.BatchNorm2d(oup, momentum=bn_momentum)

        self.relu = nn.ReLU(inplace=True)

        # self.conv11 = nn.Conv2d(dim, dim, 1) # channel mixer

    def forward(self, x):

        c33 = self.conv33(x)
        c33 = self.bn33(c33)

        inp = x

        c15 = self.conv15_1(inp)

        inp2_1 = self.bn1(inp + c15)
        c17 = self.conv17_1(inp2_1)

        inp2_2 = self.bn2(inp2_1 + c17)
        c19 = self.conv19_1(inp2_2)

        c15 = self.conv15_0(c15)

        inp2 = self.bn3(c15 + c17)
        c17 = self.conv17_0(inp2)

        inp3 = self.bn4(inp2 + c19)
        c19 = self.conv19_0(inp3)

        add = self.relu(c33 + c17 + c19 + c15)

        return add


class RPLinearBlock(nn.Module):
    def __init__(self, input_chs, ratios=[1, 0.5, 0.25], bn_momentum=0.1):
        super(RPLinearBlock, self).__init__()
        self.branches = nn.ModuleList()
        for i, ratio in enumerate(ratios):
            conv = nn.Sequential(
                nn.Conv2d(input_chs, input_chs,
                          kernel_size=(1, 2 * i + 1),
                          padding=(0, i),
                          groups=input_chs),
                nn.Conv2d(input_chs, int(input_chs * ratio),
                          kernel_size=(2 * i + 1, 1),
                          padding=(i, 0),
                          groups=int(input_chs * ratio)),
                nn.BatchNorm2d(int(input_chs * ratio), momentum=bn_momentum),
                nn.ReLU()
            )
            self.branches.append(conv)

        self.fuse_conv = nn.Sequential(  # + input_chs // 64
            nn.Conv2d(int(input_chs * sum(ratios)), input_chs,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(input_chs, momentum=bn_momentum),
            nn.ReLU()
        )

    def forward(self, x):
        branches = torch.cat([branch(x) for branch in self.branches], dim=1)
        output = self.fuse_conv(branches) + x
        return output


class L2HNet(nn.Module):
    def __init__(self,
                 width,  # width=64 for light mode; width=128 for normal mode
                 # image_band genenral is 3 (RGB) or 4 (RGB-NIR) for high-resolution remote sensing images
                 image_band=4,
                 output_chs=128,
                 length=5,
                 ratios=[1, 0.5, 0.25],
                 bn_momentum=0.1,
                 rpblock='rp',):
        super(L2HNet, self).__init__()
        self.width = width
        self.startconv = nn.Conv2d(
            image_band, self.width, kernel_size=3, stride=1, padding=1)
        self.rpblocks = nn.ModuleList()
        for _ in range(length):
            if rpblock == 'msca':
                rpblock = MSCA_SIM(self.width, self.width, bn_momentum)  # 修改
            elif rpblock == 'rpl':
                rpblock = RPLinearBlock(self.width, ratios, bn_momentum)
            else:
                rpblock = RPBlock(self.width, ratios, bn_momentum)

            self.rpblocks.append(rpblock)

        self.out_conv1 = nn.Sequential(
            StdConv2d(self.width * length, output_chs * length,
                      kernel_size=3, stride=2, bias=False, padding=1),
            nn.GroupNorm(32, output_chs*5, eps=1e-6),
            nn.ReLU()
        )
        self.out_conv2 = nn.Sequential(
            StdConv2d(output_chs * length, 1024, kernel_size=3,
                      stride=2, bias=False, padding=1),
            nn.GroupNorm(32, 1024, eps=1e-6),
            nn.ReLU()
        )
        self.out_conv3 = nn.Sequential(
            StdConv2d(1024, 1024, kernel_size=5,
                      stride=4, bias=False, padding=1),
            nn.GroupNorm(32, 1024, eps=1e-6),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.startconv(x)
        output_d1 = []
        for rpblk in self.rpblocks:
            x = rpblk(x)
            output_d1.append(x)
        output_d1 = self.out_conv1(torch.cat(output_d1, dim=1))
        output_d2 = self.out_conv2(output_d1)
        output_d3 = self.out_conv3(output_d2)
        features = [output_d1, output_d2, output_d3, x]
        return output_d3, features[::-1]
